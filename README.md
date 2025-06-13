# 🚗 AutoPark System

Sistema completo de simulación, validación y análisis predictivo de parqueaderos. Este proyecto incluye generación de datos sintéticos, herramientas de depuración visual, migración a base de datos y un agente inteligente con visualización en tiempo real.

## 🧱 Estructura del Proyecto

- `data_generator.py`: Genera archivos `.csv` simulando parqueaderos, usuarios, espacios, entradas y salidas.
- `depure_app.py`: Aplicación visual con Streamlit para verificar calidad e integridad de datos generados.
- `migrate_data.py`: Inserta los datos `.csv` en una base de datos MySQL.
- `agent_autopark.py`: Agente inteligente con predicción de ocupación y alertas en tiempo real (GUI con Tkinter).

## ⚙️ Requisitos

- Python 3.8+
- Dependencias:
```bash
pip install -r requirements.txt
````

## 🧪 1. Generar datos sintéticos

Ejecuta el generador para crear los archivos `.csv` en `generated-data/`:

```bash
python data.py
```

Archivos generados:

* `parkings.csv`
* `users.csv`
* `user_plates.csv`
* `spaces.csv`
* `entries.csv`
* `exits.csv`

## 🔍 2. Depurar los datos

Verifica que los archivos `.csv` tengan estructura y valores correctos:

```bash
streamlit run depure.py
```

Esto abrirá una interfaz web con validaciones, estadísticas, y análisis cruzado.

## 🛢️ 3. Migrar a la base de datos MySQL

Asegúrate de tener una base de datos `autopark` creada en tu servidor MySQL.

Luego ejecuta:

```bash
python migrate_data.py
```

Esto limpiará las tablas y cargará los datos desde los `.csv` respetando relaciones entre tablas.

## 🤖 4. Ejecutar el Agente Inteligente

Inicia la aplicación gráfica para predicción de ocupación y alertas:

```bash
python parking_agent/app.py
```
Running on http://127.0.0.1:5000

Características:

* Visualiza ocupación actual y futura.
* Gráfica de ocupación (últimas 24 h).
* Alerta si se predice una ocupación crítica.
* Reentrena automáticamente el modelo cada hora.

## 🛠️ Configuración

Configura la conexión a la base de datos en el archivo `agent_autopark.py`:

```python
DB_CONFIG = {
    'host':     'localhost',
    'port':     3306,
    'user':     'root',
    'password': '',
    'database': 'autopark',
    'charset':  'utf8mb4'
}
```

## 📂 Estructura de Carpetas

```
📁 generated-data/
    ├── parkings.csv
    ├── users.csv
    ├── user_plates.csv
    ├── spaces.csv
    ├── entries.csv
    └── exits.csv
```

## 📌 Notas

* El agente utiliza un modelo RandomForest para predecir ocupación.
* El sistema es extensible para múltiples parqueaderos.
* Asegúrate de tener las tablas correctas y el esquema SQL creado antes de la migración.