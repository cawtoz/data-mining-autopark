import pymysql
import pandas as pd

# Configura tus datos de conexión
db = pymysql.connect(
    host='localhost',
    user='root',
    password='',
    database='autopark',
    autocommit=True
)

cursor = db.cursor()

# Orden correcto para truncar respetando claves foráneas
tables = ["exits", "entries", "spaces", "users", "parkings"]

print("Desactivando restricciones de claves foráneas...")
cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
for table in tables:
    print(f"Vaciando tabla: {table}")
    cursor.execute(f"TRUNCATE TABLE {table}")
cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
print("Restricciones de claves foráneas reactivadas.")

# Función para cargar CSV e insertar en tabla
def load_csv_to_table(csv_filename, table_name):
    df = pd.read_csv(rf"C:\Users\cawtoz\Documents\Dev\Me\Proyecto AutoPark\data-mining\generated-data\{csv_filename}", sep=';')

    print(f"Insertando datos en {table_name} desde {csv_filename}...")

    cols = ",".join(f"`{col}`" for col in df.columns)
    placeholders = ",".join(["%s"] * len(df.columns))
    query = f"INSERT INTO `{table_name}` ({cols}) VALUES ({placeholders})"

    for _, row in df.iterrows():
        cursor.execute(query, tuple(row))

# Carga en orden respetando claves foráneas
# load_csv_to_table("vehicle_types.csv", "vehicle_types")
load_csv_to_table("parkings.csv", "parkings")
load_csv_to_table("users.csv", "users")
load_csv_to_table("spaces.csv", "spaces")
load_csv_to_table("entries.csv", "entries")
load_csv_to_table("exits.csv", "exits")

cursor.close()
db.close()
print("🚀 Migración completada con éxito.")
