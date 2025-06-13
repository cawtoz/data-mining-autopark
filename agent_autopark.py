import threading
import time
import datetime
import queue

import mysql.connector
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# --------------------------------------------
# PARÁMETROS DE CONEXIÓN A LA BASE DE DATOS
# --------------------------------------------
DB_CONFIG = {
    'host':     'localhost',      # o IP del servidor MySQL
    'port':     3306,
    'user':     'root',
    'password': '',
    'database': 'autopark',
    'charset':  'utf8mb4'
}

# --------------------------------------------
# CONSTANTES DE CONFIGURACIÓN DEL AGENTE
# --------------------------------------------
RETRAIN_INTERVAL_SECONDS = 60 * 60      # Reentrenar modelo cada 1 hora
PREDICT_AHEAD_HOURS     = 1             # Predecir ocupación 1 h en adelante
UMBRAL_ALERTA           = 0.90          # Umbral de ocupación (90 %) para disparar alerta
POLL_INTERVAL_SECONDS   = 10            # Con qué frecuencia (s) refrescar GUI y datos actuales

# --------------------------------------------
# COLA PARA COMUNICAR DATOS ENTRE HILOS
# --------------------------------------------
# Usamos una queue thread-safe para pasar:
#   (ocupacion_actual, ocupacion_predicha, lista_alertas)  →  GUI principal
data_queue = queue.Queue(maxsize=1)


class AgentPark:
    """
    Clase principal que contiene:
      - Métodos para conectar/consultar MySQL
      - Preprocesamiento de datos históricos
      - Entrenamiento y predicción del modelo
      - Lógica de alertas
    """

    def __init__(self, db_config):
        # Conexión a BD
        self.db_config = db_config
        self.conn       = None
        self._conectar_bd()

        # Modelo de Machine Learning (inicialmente None)
        self.model = None

        # Variables para guardar resultados en memoria
        self.ultimo_retrain = None
        self.ultima_prediccion = None  # float: ocupación predicha (fracción 0–1)
        self.alertas = []             # lista de cadenas (mensajes de alerta)

    def _conectar_bd(self):
        """Abre la conexión con la base de datos MySQL."""
        try:
            self.conn = mysql.connector.connect(**self.db_config)
            # Habilitamos autocommit (evita tener que hacer commit manual)
            self.conn.autocommit = True
        except Exception as e:
            print("ERROR: No se pudo conectar a la base de datos:", e)
            raise

    def _cerrar_bd(self):
        """Cierra la conexión con la base de datos (al terminar)."""
        if self.conn is not None and self.conn.is_connected():
            self.conn.close()

    # -------------------------------
    # MÓDULOS DE LECTURA DE DATOS
    # -------------------------------
    def _leer_spaces(self):
        """
        Devuelve un DataFrame con la tabla 'spaces':
         - id, is_active (BIT), is_occupied (BIT), parking_id, vehicle_type_id
        """
        query = "SELECT id, is_active, is_occupied, parking_id, vehicle_type_id FROM spaces"
        return pd.read_sql(query, con=self.conn)

    def _leer_vehicle_types(self):
        """
        Devuelve un DataFrame con la tabla 'vehicle_types':
         - id, name, hourly_rate
        """
        query = "SELECT id, name, hourly_rate FROM vehicle_types"
        return pd.read_sql(query, con=self.conn)

    def _leer_entries_exits(self):
        """
        Devuelve dos DataFrames:
          - df_entries: [id, timestamp, vehicle_owner, vehicle_plate, space_id]
          - df_exits:   [id, amount_charged, timestamp, entry_id]
        """
        df_entries = pd.read_sql("SELECT id, timestamp, space_id FROM entries", con=self.conn)
        df_exits   = pd.read_sql("SELECT e.id AS exit_id, e.timestamp AS exit_ts, e.amount_charged, "
                                 "ent.timestamp AS entry_ts, ent.space_id "
                                 "FROM exits e "
                                 "JOIN entries ent ON e.entry_id = ent.id", con=self.conn)
        return df_entries, df_exits

    # -------------------------------
    # PREPROCESAMIENTO Y FEATURES
    # -------------------------------
    def _construir_dataset_historico(self):
        """
        Construye un DataFrame que, por cada intervalo de 1 hora,
        tenga las siguientes columnas mínimas:
          - 'interval_start' (datetime, inicio de la hora)
          - 'ocupacion_actual'  (número de espacios ocupados dentro de esa hora)
          - 'ocupacion_lag_1h'  (ocupación 1 h antes)
          - 'hora_del_dia'      (0–23)
          - 'dia_semana'        (0–6, lunes=0)
          - 'ocupacion_futura'  (ocupación 1 h después, target)
        
        Para esto:
         1. Lee tabla spaces para saber el total de espacios activos.
         2. Lee entries/exits y calcula, en cada hora, cuántos vehículos entraron y salieron.
         3. A partir de ahí, calcula la ocupación neta hora a hora.
        """
        # 1) Leer histórico de entradas y salidas
        df_entries, df_exits = self._leer_entries_exits()

        # Convertir timestamps a datetime si no lo están
        df_entries['timestamp'] = pd.to_datetime(df_entries['timestamp'], errors='coerce')
        df_exits['exit_ts']     = pd.to_datetime(df_exits['exit_ts'], errors='coerce')
        df_exits['entry_ts']    = pd.to_datetime(df_exits['entry_ts'], errors='coerce')

        # 2) Crear una serie de rango horario desde la fecha mínima hasta la máxima
        min_ts = min(df_entries['timestamp'].min(), df_exits['entry_ts'].min())
        max_ts = max(df_entries['timestamp'].max(), df_exits['exit_ts'].max())
        min_hour = pd.to_datetime(min_ts).floor('h')
        max_hour = pd.to_datetime(max_ts).ceil('h')

        rango_horas = pd.date_range(start=min_hour, end=max_hour, freq='H')

        # 3) Para cada hora, contar número de entradas y salidas
        #    Entradas: las que tienen timestamp en [hora, hora+1h)
        entradas_por_hora = (df_entries
                             .assign(hour=lambda d: d['timestamp'].dt.floor('h'))
                             .groupby('hour')
                             .size()
                             .reindex(rango_horas, fill_value=0)
                             .rename('entradas'))
        salidas_por_hora = (df_exits
                            .assign(hour=lambda d: d['exit_ts'].dt.floor('h'))
                            .groupby('hour')
                            .size()
                            .reindex(rango_horas, fill_value=0)
                            .rename('salidas'))

        df_hist = pd.DataFrame({
            'interval_start': rango_horas,
            'entradas':       entradas_por_hora.values,
            'salidas':        salidas_por_hora.values
        })

        # 4) Calcular ocupación neta acumulada hora a hora:
        #    Suponemos que, al inicio, la ocupación era cero. Luego:
        #      ocupacion(t) = ocupacion(t-1) + entradas(t) - salidas(t)
        df_hist['ocupacion_actual'] = df_hist['entradas'].cumsum() - df_hist['salidas'].cumsum()
        df_hist['ocupacion_actual'] = df_hist['ocupacion_actual'].clip(lower=0)  # Evitar negativos

        # 5) Añadir features de hora del día y día de la semana
        df_hist['hora_del_dia'] = df_hist['interval_start'].dt.hour
        df_hist['dia_semana']   = df_hist['interval_start'].dt.weekday  # Lunes=0,…,Dom=6

        # 6) Para cada fila, la “ocupacion_futura” = ocupacion_actual 1 h más adelante
        df_hist['ocupacion_futura'] = df_hist['ocupacion_actual'].shift(-PREDICT_AHEAD_HOURS)
        df_hist = df_hist.dropna(subset=['ocupacion_futura']).reset_index(drop=True)

        # 7) Lag: ocupación en la hora anterior
        df_hist['ocupacion_lag_1h'] = df_hist['ocupacion_actual'].shift(1).fillna(method='bfill')

        # 8) Normalizar ocupación (opcional): 
        #    Saber cuántos espacios totales tenemos para convertir en fracción 0–1
        df_spaces = self._leer_spaces()
        total_espacios = int((df_spaces['is_active'] == 1).sum())
        df_hist['ocup_frac_actual']  = df_hist['ocupacion_actual'] / total_espacios
        df_hist['ocup_frac_futura']  = df_hist['ocupacion_futura'] / total_espacios
        df_hist['ocup_frac_lag1h']   = df_hist['ocupacion_lag_1h'] / total_espacios

        # 9) Dejamos solo columnas relevantes para el modelo
        df_features = df_hist[[
            'interval_start',
            'ocup_frac_actual',
            'ocup_frac_lag1h',
            'hora_del_dia',
            'dia_semana',
            'ocup_frac_futura'
        ]].copy()

        return df_features, total_espacios

    # -------------------------------
    # ENTRENAMIENTO Y PREDICCIÓN
    # -------------------------------
    def entrenar_modelo(self):
        """
        Lee los datos históricos, construye el dataset de features, 
        entrena un RandomForestRegressor y guarda el modelo en memoria.
        """
        df_feat, total_espacios = self._construir_dataset_historico()

        # Separar X e y
        y = df_feat['ocup_frac_futura']
        X = df_feat[['ocup_frac_actual', 'ocup_frac_lag1h', 'hora_del_dia', 'dia_semana']]

        # Dividir en train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        # Crear y entrenar RandomForest
        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        # Evaluar y mostrar MAE
        preds_test = rf.predict(X_test)
        mae = mean_absolute_error(y_test, preds_test)
        print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
              f"Modelo entrenado. MAE (fracción): {mae:.4f}")

        # Guardar en memoria
        self.model = rf
        self.total_espacios = total_espacios
        self.ultimo_retrain = datetime.datetime.now()

    def predecir_ocupacion(self):
        """
        Consulta el estado actual de 'spaces' para calcular:
          - ocup_frac_actual  = (# ocupados) / total_espacios
          - ocup_frac_lag1h   = ocup_frac 1 h atrás (usaremos el modelo para estimar)
          - hora_del_dia, dia_semana = features de tiempo actual
        Luego usa el modelo para predecir la fracción de ocupación 1 h adelante.
        """
        if self.model is None:
            return None

        # 1) Obtener el estado actual de espacios
        df_spaces = self._leer_spaces()
        total_espacios = int((df_spaces['is_active'] == 1).sum())
        actuales       = int((df_spaces['is_occupied'] == 1).sum())
        ocup_frac_act  = actuales / total_espacios

        # 2) Para el lag1h: supongamos que hace 1 h la ocupación fue similar a la última predicción
        #    (o bien podríamos almacenar la última ocup_frac_act histórica).
        if self.ultima_prediccion is not None:
            ocup_frac_lag1h = self.ultima_prediccion
        else:
            # Si nunca predijimos, asumimos lag1h = ocup_frac_act
            ocup_frac_lag1h = ocup_frac_act

        # 3) Hora y día actual
        now = datetime.datetime.now()
        hora_actual   = now.hour
        dia_semana    = now.weekday()

        # 4) Preparar features y predecir
        X_new = np.array([[
            ocup_frac_act,
            ocup_frac_lag1h,
            hora_actual,
            dia_semana
        ]])
        pred_frac = self.model.predict(X_new)[0]
        pred_frac = float(np.clip(pred_frac, 0.0, 1.0))

        # Guardar predicción y estado actual
        self.ultima_prediccion = pred_frac
        return actuales, total_espacios, pred_frac

    # -------------------------------
    # LÓGICA DE ALERTAS
    # -------------------------------
    def generar_alertas(self, actuales, total_espacios, pred_frac):
        """
        Con base en la predicción, genera mensajes de alerta si se superan umbrales.
        """
        self.alertas.clear()

        umbral_abs = UMBRAL_ALERTA * total_espacios

        # Alerta 1: si la ocupación actual ya está sobre el umbral (e.g. >90%)
        if actuales >= umbral_abs:
            msg = (f"ALERTA: ocupación actual ALTA → "
                   f"{actuales}/{total_espacios} "
                   f"({actuales/total_espacios:.0%}).")
            self.alertas.append(msg)

        # Alerta 2: si la predicción 1 h adelante indica ocupación sobre el umbral
        pred_abs = pred_frac * total_espacios
        if pred_abs >= umbral_abs:
            msg = (f"ALERTA: en 1 h PREDICCIÓN de alta ocupación → "
                   f"{int(pred_abs)}/{total_espacios} "
                   f"({pred_frac:.0%}).")
            self.alertas.append(msg)

        # Alerta 3: cambio brusco en predicción vs actual (>10% de diferencia)
        diff = abs(pred_frac - (actuales / total_espacios))
        if diff >= 0.10:
            msg = (f"ADVERTENCIA: cambio brusco detectado: "
                   f"ocup actual {actuales/total_espacios:.0%}, "
                   f"predicción {pred_frac:.0%}.")
            self.alertas.append(msg)

        return list(self.alertas)

    # -------------------------------
    # HILO DE EJECUCIÓN PERIÓDICA
    # -------------------------------
    def run_periodic(self):
        """
        Función que corre en un thread aparte:
         1) Reentrena el modelo al inicio y cada RETRAIN_INTERVAL_SECONDS.
         2) Cada vez que se tenga modelo, hace predicción y genera alertas.
         3) Encola (ocup_actual, total_espacios, pred_frac, alertas) para la GUI.
         4) Duerme POLL_INTERVAL_SECONDS y repite.
        """
        # 1) Entrenamiento inicial
        try:
            self.entrenar_modelo()
        except Exception as e:
            print("Error entrenando modelo inicial:", e)
            return

        next_retrain = time.time() + RETRAIN_INTERVAL_SECONDS

        while True:
            # 2) Comprobar si toca reentrenar
            if time.time() >= next_retrain:
                try:
                    self.entrenar_modelo()
                except Exception as e:
                    print("Error reentrenando modelo:", e)
                next_retrain = time.time() + RETRAIN_INTERVAL_SECONDS

            # 3) Si hay modelo, predecir y generar alertas
            try:
                res = self.predecir_ocupacion()
                if res is not None:
                    actuales, total_esp, pred_frac = res
                    alertas = self.generar_alertas(actuales, total_esp, pred_frac)
                    payload = {
                        'actuales':     actuales,
                        'total_esp':    total_esp,
                        'pred_frac':    pred_frac,
                        'alertas':      alertas,
                        'timestamp':    datetime.datetime.now()
                    }
                    # Enviar a la cola (si está llena, sobrescribimos el anterior)
                    if data_queue.full():
                        try:
                            data_queue.get_nowait()
                        except queue.Empty:
                            pass
                    data_queue.put(payload)
            except Exception as e:
                print("Error en predicción o alertas:", e)

            # 4) Dormir antes de la próxima iteración
            time.sleep(POLL_INTERVAL_SECONDS)


# --------------------------------------------
# INTERFAZ GRÁFICA CON TKINTER
# --------------------------------------------
class AgentGUI(tk.Tk):
    """
    Ventana principal donde:
     - Se muestran contadores de ocupación actual y predicha.
     - Una gráfica de ocupación/alertas.
     - Un Treeview con historial de alertas.
    """

    def __init__(self, agent: AgentPark):
        super().__init__()

        self.agent = agent
        self.title("Agent Autopark")
        self.geometry("800x600")
        self.resizable(False, False)

        # Frames principales
        self.frame_status = ttk.LabelFrame(self, text="Estado en Tiempo Real")
        self.frame_status.place(x=10, y=10, width=380, height=120)

        self.frame_pred   = ttk.LabelFrame(self, text="Predicción a 1 h")
        self.frame_pred.place(x=410, y=10, width=380, height=120)

        self.frame_alert  = ttk.LabelFrame(self, text="Alertas Recientes")
        self.frame_alert.place(x=10, y=140, width=780, height=200)

        self.frame_plot   = ttk.LabelFrame(self, text="Gráfica de Ocupación")
        self.frame_plot.place(x=10, y=350, width=780, height=240)

        # Labels para estado actual
        self.lbl_actuales_var = tk.StringVar(value="Cargando...")
        ttk.Label(self.frame_status, text="Vehículos Ocupando:").place(x=10, y=10)
        ttk.Label(self.frame_status, textvariable=self.lbl_actuales_var, font=("Arial", 14, "bold")).place(x=10, y=40)

        self.lbl_total_var = tk.StringVar(value="Cargando...")
        ttk.Label(self.frame_status, text="Total Espacios:").place(x=200, y=10)
        ttk.Label(self.frame_status, textvariable=self.lbl_total_var, font=("Arial", 14, "bold")).place(x=200, y=40)

        # Labels para predicción
        self.lbl_pred_var = tk.StringVar(value="Cargando...")
        ttk.Label(self.frame_pred, text="Ocup. estimada 1 h:").place(x=10, y=10)
        ttk.Label(self.frame_pred, textvariable=self.lbl_pred_var, font=("Arial", 14, "bold")).place(x=10, y=40)

        # Treeview para alertas
        cols = ("timestamp", "mensaje")
        self.tree_alertas = ttk.Treeview(self.frame_alert, columns=cols, show='headings')
        self.tree_alertas.heading("timestamp", text="Fecha / Hora")
        self.tree_alertas.heading("mensaje",   text="Mensaje de Alerta")
        self.tree_alertas.column("timestamp", width=180, anchor="center")
        self.tree_alertas.column("mensaje",   width=570, anchor="w")
        self.tree_alertas.place(x=10, y=10, width=760, height=180)
        self.scrollbar = ttk.Scrollbar(self.frame_alert, orient="vertical", command=self.tree_alertas.yview)
        self.tree_alertas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.place(x=770, y=10, height=180)

        # Gráfica de ocupación histórica y predicción (placeholder vacío)
        self.fig, self.ax = plt.subplots(figsize=(7.5, 2.8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_plot)
        self.canvas.get_tk_widget().place(x=10, y=10, width=760, height=220)
        self.ax.set_title("Ocupación Histórica (últimas 24 h)")
        self.ax.set_xlabel("Hora")
        self.ax.set_ylabel("Ocupación (número de vehículos)")

        # Datos para graficar
        self.hist_timestamps = []
        self.hist_ocupacion   = []

        # Iniciar actualización periódica de la GUI
        self.after(1000, self.actualizar_gui)

    def actualizar_gui(self):
        """
        Se llama cada POLL_INTERVAL_SECONDS (aprox) para:
          - Tomar el último payload de data_queue
          - Actualizar labels de estado y predicción
          - Insertar nuevas alertas en el Treeview
          - Actualizar la gráfica de ocupación (últimas 24 h)
        """
        try:
            payload = data_queue.get_nowait()
        except queue.Empty:
            # Si no hay datos nuevos, volver a intentar más tarde
            self.after(POLL_INTERVAL_SECONDS * 1000, self.actualizar_gui)
            return

        # Desempaquetar payload
        actuales   = payload['actuales']
        total_esp  = payload['total_esp']
        pred_frac  = payload['pred_frac']
        alertas    = payload['alertas']
        ts         = payload['timestamp']

        # 1) Actualizar labels
        self.lbl_actuales_var.set(f"{actuales}")
        self.lbl_total_var.set(f"{total_esp}")
        self.lbl_pred_var.set(f"{int(pred_frac * total_esp)}/{total_esp} "
                               f"({pred_frac:.0%})")

        # 2) Agregar alertas al Treeview (solo las nuevas)
        for msg in alertas:
            # Evitar duplicados: chequeamos si ya existe
            existe = False
            for child in self.tree_alertas.get_children():
                if self.tree_alertas.item(child, "values")[1] == msg:
                    existe = True
                    break
            if not existe:
                self.tree_alertas.insert("", "end",
                                         values=(ts.strftime("%Y-%m-%d %H:%M:%S"), msg))

        # 3) Actualizar datos para la gráfica: mantenemos histórico de últimas 24 h
        self.hist_timestamps.append(ts)
        self.hist_ocupacion.append(actuales)
        # Filtrar solo timestamps > 24 h atrás
        cutoff = ts - datetime.timedelta(hours=24)
        while self.hist_timestamps and self.hist_timestamps[0] < cutoff:
            self.hist_timestamps.pop(0)
            self.hist_ocupacion.pop(0)

        # 4) Dibujar la gráfica actualizada
        self.ax.clear()
        self.ax.plot(self.hist_timestamps, self.hist_ocupacion, marker='o', linestyle='-')
        self.ax.set_title("Ocupación Histórica (últimas 24 h)")
        self.ax.set_xlabel("Hora")
        self.ax.set_ylabel("N.º de Vehículos")
        self.ax.tick_params(axis='x', rotation=45)
        self.fig.tight_layout()
        self.canvas.draw()

        # Programar próxima actualización
        self.after(POLL_INTERVAL_SECONDS * 1000, self.actualizar_gui)


# --------------------------------------------
# FUNCIÓN PRINCIPAL
# --------------------------------------------
def main():
    # 1) Inicializar agente
    agent = AgentPark(DB_CONFIG)

    # 2) Iniciar hilo de ejecución periódica
    hilo = threading.Thread(target=agent.run_periodic, daemon=True)
    hilo.start()

    # 3) Iniciar GUI
    app = AgentGUI(agent)
    app.mainloop()

    # 4) Al cerrar GUI, cerramos conexión a BD
    agent._cerrar_bd()


if __name__ == "__main__":
    main()
