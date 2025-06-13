import os
import random
import logging
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("realistic-gen")

NUM_PARKINGS = 5
SPACES_PER = 10
NUM_USERS = 20
NUM_EVENTS = 60000
ROLES = ["USER", "ADMIN"]
# Tarifas por hora según tipo de vehículo (ejemplo: 1: auto, 2: moto, etc)
VEHICLE_TYPES = {1: 1.0, 2: 2.0, 3: 0.5, 4: 4.0}  # tarifa por hora
OUTDIR = "generated-data"

fake = Faker()
Faker.seed(42)
random.seed(42)

os.makedirs(OUTDIR, exist_ok=True)

def save_df(name: str, df: pd.DataFrame):
    path = os.path.join(OUTDIR, f"{name}.csv")
    df.to_csv(path, sep=";", index=False)
    logger.info(f"{name}.csv → {len(df)} registros")

logger.info("→ Generando parqueaderos...")
parkings = []
for i in range(1, NUM_PARKINGS + 1):
    city = fake.city()
    address = fake.address().replace("\n", ", ").replace("  ", " ")
    name = f"Parqueadero {city} {i}"
    parkings.append({"id": i, "name": name, "address": address})
df_parkings = pd.DataFrame(parkings)
save_df("parkings", df_parkings)

logger.info("→ Generando usuarios y placas...")
users, plates = [], []
for i in range(1, NUM_USERS + 1):
    fn = fake.first_name()
    ln = fake.last_name()
    username = (fn[0] + ln).lower()
    email = f"{username}@example.com"
    role = random.choices(ROLES, weights=[0.85, 0.15])[0]
    phone = fake.msisdn()[:10]
    parking_id = random.randint(1, NUM_PARKINGS)
    password = fake.password(length=12)

    users.append({
        "id": i, "email": email, "name": fn, "surname": ln,
        "username": username, "password": password,
        "phone": phone, "role": role, "parking_id": parking_id
    })

    for _ in range(random.randint(1, 2)):
        letters = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=3))
        numbers = ''.join(random.choices('0123456789', k=3))
        plates.append({"user_id": i, "plate": f"{letters}-{numbers}"})
df_users = pd.DataFrame(users)
df_plates = pd.DataFrame(plates)
save_df("users", df_users)
save_df("user_plates", df_plates)

logger.info("→ Generando espacios...")
spaces = []
space_prefixes = ['A', 'B', 'C', 'P']
vehicle_type_ids = list(VEHICLE_TYPES.keys())
sid = 1
for pid in range(1, NUM_PARKINGS + 1):
    for j in range(1, SPACES_PER + 1):
        name = f"{random.choice(space_prefixes)}{j:02d}"
        vehicle_type_id = random.choices(vehicle_type_ids, weights=[0.5, 0.3, 0.1, 0.1])[0]
        spaces.append({
            "id": sid, "is_active": 1, "is_occupied": 0,
            "name": name, "parking_id": pid,
            "vehicle_type_id": vehicle_type_id
        })
        sid += 1
df_spaces = pd.DataFrame(spaces)
save_df("spaces", df_spaces)

logger.info("→ Simulando entradas y salidas con cargos...")
start_time = datetime.now() - timedelta(days=30)
end_time = datetime.now()

entries, exits = [], []
occupied = {s: None for s in df_spaces['id']}
entry_id = 1
exit_id = 1

space_ids = list(df_spaces['id'])
plate_list = df_plates['plate'].tolist()

# Prepara para búsqueda de tipo de vehículo por espacio_id
space_vehicle_map = df_spaces.set_index('id')['vehicle_type_id'].to_dict()

# Generar timestamps ordenados para eventos
timestamps = sorted([
    start_time + timedelta(minutes=random.randint(0, 60 * 24 * 30))
    for _ in range(NUM_EVENTS)
])

def calcular_cargo(entry_time, exit_time, rate_per_hour):
    tiempo = exit_time - entry_time
    horas = tiempo.total_seconds() / 3600
    return round(max(horas, 0.25) * rate_per_hour, 2)  # mínimo cobro 15 min (0.25h)

for ts in timestamps:
    # 70% probabilidad de entrada si hay espacio libre
    if random.random() < 0.7 and any(v is None for v in occupied.values()):
        free_spaces = [sid for sid, val in occupied.items() if val is None]
        sid = random.choice(free_spaces)
        plate = random.choice(plate_list)
        owner = fake.name()

        entries.append({
            "id": entry_id,
            "timestamp": ts,
            "vehicle_owner": owner,
            "vehicle_plate": plate,
            "space_id": sid
        })

        occupied[sid] = (entry_id, ts)
        entry_id += 1

    # 30% probabilidad de salida si hay espacio ocupado
    elif any(v is not None for v in occupied.values()):
        sid = random.choice([s for s, v in occupied.items() if v is not None])
        current_entry_id, entry_ts = occupied[sid]

        # La salida ocurre entre 1 min y 6 horas después de la entrada
        delta = timedelta(minutes=random.randint(1, 360))
        exit_ts = entry_ts + delta

        if exit_ts > end_time:
            exit_ts = end_time

        # Calcular cargo según tarifa del tipo de vehículo de ese espacio
        vehicle_type = space_vehicle_map[sid]
        amount_charge = calcular_cargo(entry_ts, exit_ts, VEHICLE_TYPES[vehicle_type])

        exits.append({
            "id": exit_id,
            "timestamp": exit_ts,
            "entry_id": current_entry_id,
            "amount_charged": amount_charge
        })

        occupied[sid] = None
        exit_id += 1

df_entries = pd.DataFrame(entries)
df_exits = pd.DataFrame(exits)

save_df("entries", df_entries)
save_df("exits", df_exits)

logger.info("→ Actualizando estado final de espacios...")
df_entries['timestamp_dt'] = pd.to_datetime(df_entries['timestamp'])
df_exits['timestamp_dt'] = pd.to_datetime(df_exits['timestamp'])

for idx, row in df_spaces.iterrows():
    sid = row['id']
    ents = df_entries[df_entries['space_id'] == sid]
    exs = df_exits.merge(df_entries[['id', 'space_id']], left_on='entry_id', right_on='id', how='left')
    exs = exs[exs['space_id'] == sid]
    latest_ent = ents['timestamp_dt'].max() if not ents.empty else None
    latest_ext = exs['timestamp_dt'].max() if not exs.empty else None
    df_spaces.at[idx, 'is_occupied'] = int(
        latest_ent is not None and (latest_ext is None or latest_ent > latest_ext)
    )

save_df("spaces", df_spaces)

print("\nResumen Final")
print("-------------")
print(f"Parkings: {len(parkings)}")
print(f"Users: {len(users)}")
print(f"Placas: {len(plates)}")
print(f"Spaces: {len(spaces)}")
print(f"Entries: {len(entries)}")
print(f"Exits: {len(exits)}")
logger.info("✅ Generación de datos completada.")
