import os
import pandas as pd
import streamlit as st
from datetime import datetime

OUTDIR = "generated-data"

EXPECTED_SCHEMAS = {
    "parkings": ["id", "name", "address"],
    "users": ["id", "email", "name", "surname", "username", "password", "phone", "role", "parking_id"],
    "spaces": ["id", "is_active", "is_occupied", "name", "parking_id", "vehicle_type_id"],
    "entries": ["id", "timestamp", "owner", "plate", "space_id"],
    "exits": ["id", "timestamp", "entry_id", "space_id"]
}

VALID_ROLES = {"ADMIN", "USER"}

def load_csv(name):
    path = os.path.join(OUTDIR, f"{name}.csv")
    return pd.read_csv(path, sep=";") if os.path.exists(path) else None

st.title("Depuración de Datos - Parking System")

data = {table: load_csv(table) for table in EXPECTED_SCHEMAS}
now = datetime.now()

for table_name, expected_fields in EXPECTED_SCHEMAS.items():
    st.header(f"Tabla: {table_name}")

    df = data[table_name]
    if df is None:
        st.error(f"❌ Archivo {table_name}.csv no encontrado en {OUTDIR}/")
        continue

    st.write(f"📊 Registros cargados: {len(df)}")

    # Verificar columnas esperadas
    missing_fields = [f for f in expected_fields if f not in df.columns]
    extra_fields = [f for f in df.columns if f not in expected_fields]

    if missing_fields:
        st.error(f"❌ Faltan campos: {missing_fields}")
    else:
        st.success("✅ Todos los campos esperados están presentes")

    if extra_fields:
        st.warning(f"⚠️ Campos inesperados: {extra_fields}")
    else:
        st.success("✅ No hay campos inesperados")

    st.subheader("Primeras 5 filas")
    st.dataframe(df.head(5))

    # Validaciones generales
    st.subheader("Validaciones de Calidad")

    any_quality_checks = False
    for col in expected_fields:
        if col not in df.columns:
            continue

        # Nulls
        nulls = df[col].isnull().sum()
        if nulls > 0:
            any_quality_checks = True
            st.warning(f"⚠️ {col}: {nulls} valores nulos")
            st.dataframe(df[df[col].isnull()].head(5))
        else:
            st.success(f"✅ {col}: no hay valores nulos")

        # Vacíos / en blanco (para strings)
        if df[col].dtype == "object":
            empty_like = df[col].apply(lambda x: str(x).strip() == "").sum()
            if empty_like > 0:
                any_quality_checks = True
                st.warning(f"⚠️ {col}: {empty_like} valores vacíos o en blanco")
                st.dataframe(df[df[col].apply(lambda x: str(x).strip() == "")].head(5))
            else:
                st.success(f"✅ {col}: no hay valores vacíos")

            short_values = df[col].apply(lambda x: isinstance(x, str) and len(x.strip()) < 2).sum()
            if short_values > 0:
                any_quality_checks = True
                st.warning(f"⚠️ {col}: {short_values} valores con longitud < 2")
                st.dataframe(df[df[col].apply(lambda x: isinstance(x, str) and len(x.strip()) < 2)].head(5))
            else:
                st.success(f"✅ {col}: no hay valores demasiado cortos")

    if not any_quality_checks:
        st.success("✅ Todas las validaciones de calidad pasaron sin incidencias")

    # Validación de campos booleanos
    if table_name == "spaces":
        any_bool_bad = False
        for field in ["is_active", "is_occupied"]:
            if field in df.columns:
                bad_values = df[~df[field].isin([0, 1, True, False])]
                if not bad_values.empty:
                    any_bool_bad = True
                    st.error(f"❌ {field}: {len(bad_values)} valores no válidos")
                    st.dataframe(bad_values.head(5))
                else:
                    st.success(f"✅ {field}: todos los valores son 0/1 o True/False")
        if not any_bool_bad:
            st.success("✅ Validación de campos booleanos completada sin errores")

    # Validación de roles
    if table_name == "users" and "role" in df.columns:
        invalid_roles = df[~df["role"].isin(VALID_ROLES)]
        if not invalid_roles.empty:
            st.warning(f"⚠️ {len(invalid_roles)} roles no válidos detectados")
            st.dataframe(invalid_roles[["id", "email", "role"]].head(5))
        else:
            st.success("✅ Todos los roles son válidos")

    # Validación de fechas
    if table_name in {"entries", "exits"} and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        future_dates = df[df["timestamp"] > now]
        if not future_dates.empty:
            st.warning(f"⚠️ {len(future_dates)} registros con fecha futura")
            st.dataframe(future_dates.head(5))
        else:
            st.success("✅ No hay fechas futuras en los timestamps")

        invalid_dates = df["timestamp"].isna().sum()
        if invalid_dates > 0:
            st.error(f"❌ {invalid_dates} timestamps no válidos")
            st.dataframe(df[df["timestamp"].isna()].head(5))
        else:
            st.success("✅ Todos los timestamps son válidos")

    st.markdown("---")

# Validación de relaciones cruzadas: entries vs exits
if data["entries"] is not None and data["exits"] is not None:
    st.header("Validaciones entre Entradas y Salidas")
    entries = data["entries"].copy()
    exits = data["exits"].copy()

    entries["timestamp"] = pd.to_datetime(entries["timestamp"], errors="coerce")
    exits["timestamp"] = pd.to_datetime(exits["timestamp"], errors="coerce")

    merged = exits.merge(entries, left_on="entry_id", right_on="id", suffixes=("_exit", "_entry"))
    invalid_duration = merged[merged["timestamp_exit"] < merged["timestamp_entry"]]
    if not invalid_duration.empty:
        st.error(f"❌ {len(invalid_duration)} salidas ocurren antes que su entrada")
        st.dataframe(invalid_duration[["id_exit", "timestamp_exit", "entry_id", "timestamp_entry", "space_id_exit"]].head(5))
    else:
        st.success("✅ Todas las salidas ocurren después de su entrada correspondiente")

# Validación de espacios ocupados
if data["spaces"] is not None and data["entries"] is not None and data["exits"] is not None:
    st.header("Validación de Ocupación de Espacios")
    entries = data["entries"].copy()
    exits = data["exits"].copy()
    spaces = data["spaces"].copy()

    entries["timestamp"] = pd.to_datetime(entries["timestamp"], errors="coerce")
    exits["timestamp"] = pd.to_datetime(exits["timestamp"], errors="coerce")

    last_entries = entries.groupby("space_id")["timestamp"].max().reset_index()
    last_exits = exits.groupby("space_id")["timestamp"].max().reset_index()

    merged = pd.merge(last_entries, last_exits, on="space_id", how="left", suffixes=("_entry", "_exit"))
    inconsistents = merged[merged["timestamp_exit"].isna() | (merged["timestamp_entry"] > merged["timestamp_exit"])]

    st.write(f"🔍 Espacios con entrada sin salida posterior: {len(inconsistents)}")
    if not inconsistents.empty:
        st.dataframe(inconsistents.head(5))
    else:
        st.success("✅ No hay inconsistencias entre entradas y salidas por espacio")

    actual_occupied = set(spaces[spaces["is_occupied"].isin([1, True])]["id"])
    expected_occupied = set(inconsistents["space_id"])
    mismatch = actual_occupied.symmetric_difference(expected_occupied)

    if mismatch:
        st.warning(f"⚠️ {len(mismatch)} espacios con ocupación incoherente respecto a entradas/salidas")
        st.write("IDs de espacios con discrepancias (ejemplo):")
        st.write(list(mismatch)[:10])
    else:
        st.success("✅ La ocupación de espacios coincide con entradas y salidas")

# Validación de claves foráneas
st.header("Validación de Relaciones entre Tablas (FKs)")
# spaces ↔ parkings
if data["spaces"] is not None and data["parkings"] is not None:
    invalid_spaces = data["spaces"][~data["spaces"]["parking_id"].isin(data["parkings"]["id"])]
    if not invalid_spaces.empty:
        st.error(f"❌ {len(invalid_spaces)} espacios tienen parking_id no existente")
        st.dataframe(invalid_spaces.head(5))
    else:
        st.success("✅ Todos los parking_id en spaces son válidos")

# users ↔ parkings
if data["users"] is not None and data["parkings"] is not None:
    invalid_users = data["users"][~data["users"]["parking_id"].isin(data["parkings"]["id"])]
    if not invalid_users.empty:
        st.error(f"❌ {len(invalid_users)} usuarios tienen parking_id no existente")
        st.dataframe(invalid_users.head(5))
    else:
        st.success("✅ Todos los parking_id en users son válidos")

# entries ↔ spaces
if data["entries"] is not None and data["spaces"] is not None:
    invalid_entries = data["entries"][~data["entries"]["space_id"].isin(data["spaces"]["id"])]
    if not invalid_entries.empty:
        st.error(f"❌ {len(invalid_entries)} entradas con space_id inválido")
        st.dataframe(invalid_entries.head(5))
    else:
        st.success("✅ Todos los space_id en entries son válidos")

# exits ↔ entries
if data["exits"] is not None and data["entries"] is not None:
    invalid_exits = data["exits"][~data["exits"]["entry_id"].isin(data["entries"]["id"])]
    if not invalid_exits.empty:
        st.error(f"❌ {len(invalid_exits)} salidas con entry_id inválido")
        st.dataframe(invalid_exits.head(5))
    else:
        st.success("✅ Todos los entry_id en exits son válidos")

# Análisis por parking
if data["spaces"] is not None and data["parkings"] is not None:
    st.header("Análisis por Parking")
    summary = data["spaces"].groupby("parking_id").agg(
        total_spaces=("id", "count"),
        occupied=("is_occupied", lambda x: (x == 1).sum())
    )
    summary["libres"] = summary["total_spaces"] - summary["occupied"]
    st.dataframe(summary)

st.success("🎉 Depuración completada.")
