import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import DateFormatter

# Configuración general de estilo para seaborn y matplotlib
sns.set(style="whitegrid", palette="muted")
plt.rcParams.update({'axes.titlesize': 16, 'axes.labelsize': 14, 'xtick.labelsize': 12, 'ytick.labelsize': 12})

# Directorio de datos
OUTDIR = "generated-data"

# Lectura de archivos CSV con manejo de errores y dtype optimizado
def load_data(file_name):
    path = os.path.join(OUTDIR, file_name)
    try:
        return pd.read_csv(path, sep=";")
    except FileNotFoundError:
        print(f"Archivo no encontrado: {file_name}")
        return pd.DataFrame()

df_spaces = load_data('spaces.csv')
df_entries = load_data('entries.csv')
df_exits = load_data('exits.csv')

# Conversiones de fechas para entries y exits
for df in [df_entries, df_exits]:
    if not df.empty and 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)  # quitar filas con fechas inválidas
        df['date'] = df['timestamp'].dt.date

# Funciones para gráficos mejorados

def grafico_1():
    """
    Estado actual de los espacios:
    - Muestra proporción con % exacto y número total.
    - Usa colores amigables y leyenda detallada.
    """
    if df_spaces.empty:
        print("No hay datos para espacios.")
        return
    
    counts = df_spaces['is_occupied'].value_counts().sort_index()
    labels = ['Libres', 'Ocupados']
    colors = ['#4CAF50', '#E53935']  # Verde y rojo más suaves
    
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        autopct=lambda p: f"{p:.1f}%\n({int(p*sum(counts)/100)})",
        startangle=140,
        colors=colors,
        textprops={'fontsize': 14, 'weight': 'bold'}
    )
    ax.set_title('Estado Actual de los Espacios', pad=20)
    plt.show()

def grafico_2():
    """
    Entradas por día:
    - Lineplot con puntos destacados.
    - Formato de fecha legible y grid.
    - Estadísticas de resumen (media, mediana).
    """
    if df_entries.empty:
        print("No hay datos para entradas.")
        return
    
    entries_per_day = df_entries.groupby('date').size()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.lineplot(x=entries_per_day.index, y=entries_per_day.values, marker='o', ax=ax, color='#1f77b4')
    ax.set_title("Entradas Diarias", fontsize=16, weight='bold')
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Número de entradas")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Agregar líneas de estadística
    media = entries_per_day.mean()
    mediana = entries_per_day.median()
    ax.axhline(media, color='orange', linestyle='--', label=f'Media: {media:.1f}')
    ax.axhline(mediana, color='green', linestyle='-.', label=f'Mediana: {mediana}')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def grafico_3():
    """
    Distribución de tipos de vehículos:
    - Barra horizontal para mejor lectura.
    - Colores gradientes y etiquetas con porcentaje.
    """
    if df_spaces.empty or 'vehicle_type_id' not in df_spaces.columns:
        print("No hay datos para tipos de vehículos.")
        return
    
    vehicle_counts = df_spaces['vehicle_type_id'].value_counts(normalize=True).sort_index()
    vehicle_labels = [f"Tipo {k}" for k in vehicle_counts.index]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=vehicle_counts.values, y=vehicle_labels, palette="Blues_d", ax=ax)
    
    ax.set_title("Distribución de Tipos de Vehículos", fontsize=16, weight='bold')
    ax.set_xlabel("Porcentaje del total (%)")
    ax.set_xlim(0, vehicle_counts.max() * 1.1)
    
    # Añadir porcentaje en las barras
    for i, val in enumerate(vehicle_counts.values):
        ax.text(val + 0.01, i, f"{val*100:.1f}%", va='center', fontsize=12)
    
    plt.tight_layout()
    plt.show()

def grafico_4():
    """
    Entradas vs Salidas por día:
    - Gráfico de barras agrupadas con colores definidos.
    - Mostrar total acumulado y diferencias.
    """
    if df_entries.empty or df_exits.empty:
        print("No hay datos suficientes para entradas y salidas.")
        return
    
    entries_per_day = df_entries.groupby('date').size()
    exits_per_day = df_exits.groupby('date').size()
    combined = pd.DataFrame({'Entradas': entries_per_day, 'Salidas': exits_per_day}).fillna(0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    combined.plot(kind='bar', stacked=False, color=["#4caf50", "#f44336"], ax=ax)
    
    ax.set_title("Entradas vs Salidas por Día", fontsize=16, weight='bold')
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Cantidad")
    plt.xticks(rotation=45)
    
    # Añadir total diario encima de las barras
    for i, (entrada, salida) in enumerate(zip(combined['Entradas'], combined['Salidas'])):
        total = entrada + salida
        ax.text(i, max(entrada, salida) + 1, f'Total: {int(total)}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def grafico_5():
    """
    Estadísticas adicionales:
    - Duración promedio de estancia por vehículo (si los datos permiten).
    - Análisis de tiempo entre entrada y salida.
    """
    # Solo si df_entries y df_exits tienen columna 'vehicle_id' y 'timestamp'
    if df_entries.empty or df_exits.empty:
        print("Datos insuficientes para análisis de duración.")
        return
    
    # Asumimos que hay una columna 'vehicle_id' para cruzar datos
    if 'vehicle_id' not in df_entries.columns or 'vehicle_id' not in df_exits.columns:
        print("Faltan columnas necesarias para análisis de duración.")
        return
    
    # Merge por vehicle_id para unir entrada y salida
    df_merge = pd.merge(df_entries[['vehicle_id', 'timestamp']], df_exits[['vehicle_id', 'timestamp']], on='vehicle_id', suffixes=('_entry', '_exit'))
    
    # Filtrar salidas después de entradas (caso normal)
    df_merge = df_merge[df_merge['timestamp_exit'] > df_merge['timestamp_entry']]
    
    # Calcular duración en minutos
    df_merge['duracion_minutos'] = (df_merge['timestamp_exit'] - df_merge['timestamp_entry']).dt.total_seconds() / 60
    
    # Gráfico distribución de duración
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_merge['duracion_minutos'], bins=30, kde=True, color='#2196f3', ax=ax)
    ax.set_title("Distribución de Duración de Estancia (minutos)", fontsize=16, weight='bold')
    ax.set_xlabel("Minutos")
    ax.set_ylabel("Frecuencia")
    
    # Estadísticas
    media = df_merge['duracion_minutos'].mean()
    mediana = df_merge['duracion_minutos'].median()
    ax.axvline(media, color='orange', linestyle='--', label=f'Media: {media:.1f} min')
    ax.axvline(mediana, color='green', linestyle='-.', label=f'Mediana: {mediana:.1f} min')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

# Interfaz gráfica con Tkinter mejorada
root = tk.Tk()
root.title("Visualización de Datos de Estacionamiento")
root.geometry("450x350")
root.resizable(False, False)

# Título y estilo
label = ttk.Label(root, text="Selecciona un gráfico para visualizar:", font=("Arial", 16, "bold"))
label.pack(pady=25)

# Botones con padding y tamaño uniforme
botones = [
    ("Estado actual de los espacios", grafico_1),
    ("Entradas por día", grafico_2),
    ("Tipos de vehículos", grafico_3),
    ("Entradas vs Salidas", grafico_4),
    ("Duración promedio de estancia", grafico_5)
]

for texto, funcion in botones:
    ttk.Button(root, text=texto, command=funcion).pack(pady=8, ipadx=12, ipady=6)

root.mainloop()
