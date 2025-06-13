import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import datetime
import time
import queue
import threading
import logging # Usar logging para una mejor gestión de mensajes

from database import DatabaseManager, DB_CONFIG

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# Agent configuration constants
RETRAIN_INTERVAL_SECONDS = 60 * 60  # Retrain model every 1 hour (3600 seconds)
PREDICT_AHEAD_HOURS = 1             # Predict occupation 1 hour ahead
OCCUPANCY_ALERT_THRESHOLD = 0.90    # Occupation threshold (90%) for critical alerts
POLL_INTERVAL_SECONDS = 10          # How often (s) to refresh data and make predictions

class AgentPark:
    """
    Intelligent agent for a single parking lot.
    Manages data retrieval, model training, prediction, and alert generation.
    """
    def __init__(self, parking_id: int):
        self.parking_id = parking_id
        self.db_manager = DatabaseManager(DB_CONFIG)

        self.model = None  # Machine Learning model
        self.total_parking_spaces = 0
        self.last_retrain_time = None
        self.last_prediction_value = None  # float: predicted occupation (0-1 fraction)
        self.current_alerts = []       # List of alert messages (dict with 'type' and 'message')

        # Queue for sending data to the web interface (thread-safe)
        self.data_queue = queue.Queue(maxsize=1)

    def _build_historical_dataset(self) -> tuple[pd.DataFrame, int]:
        """
        Builds a historical dataset for a specific parking lot,
        calculating hourly occupation and relevant features.
        """
        logging.info(f"Parking ID {self.parking_id}: Building historical dataset.")
        df_entries, df_exits = self.db_manager.get_entries_exits(self.parking_id)

        df_entries['timestamp'] = pd.to_datetime(df_entries['timestamp'], errors='coerce')
        df_exits['exit_ts'] = pd.to_datetime(df_exits['exit_ts'], errors='coerce')
        df_exits['entry_ts'] = pd.to_datetime(df_exits['entry_ts'], errors='coerce')

        df_entries.dropna(subset=['timestamp'], inplace=True)
        df_exits.dropna(subset=['exit_ts', 'entry_ts'], inplace=True)

        if df_entries.empty and df_exits.empty:
            logging.warning(f"Parking ID {self.parking_id}: No historical data for dataset build.")
            return pd.DataFrame(), 0

        # Determine overall time range from all relevant timestamps
        all_timestamps = pd.Series([])
        if not df_entries.empty:
            all_timestamps = pd.concat([all_timestamps, df_entries['timestamp']])
        if not df_exits.empty:
            all_timestamps = pd.concat([all_timestamps, df_exits['entry_ts'], df_exits['exit_ts']])

        if all_timestamps.empty:
            logging.warning(f"Parking ID {self.parking_id}: No valid timestamps to build dataset.")
            return pd.DataFrame(), 0

        min_ts = all_timestamps.min()
        max_ts = all_timestamps.max()

        if pd.isna(min_ts) or pd.isna(max_ts):
            logging.warning(f"Parking ID {self.parking_id}: Insufficient valid time data for dataset build.")
            return pd.DataFrame(), 0

        min_hour = pd.to_datetime(min_ts).floor('h')
        max_hour = pd.to_datetime(max_ts).ceil('h')
        
        if min_hour == max_hour:
            max_hour = min_hour + pd.Timedelta(hours=1)

        hourly_range = pd.date_range(start=min_hour, end=max_hour, freq='H')
        if hourly_range.empty: # Handle case where only one timestamp might exist
            logging.warning(f"Parking ID {self.parking_id}: Hourly range is empty. Not enough data points.")
            return pd.DataFrame(), 0


        entries_per_hour = (df_entries
                            .assign(hour=lambda d: d['timestamp'].dt.floor('h'))
                            .groupby('hour')
                            .size()
                            .reindex(hourly_range, fill_value=0)
                            .rename('entradas'))
        
        exits_per_hour = (df_exits
                          .assign(hour=lambda d: d['exit_ts'].dt.floor('h'))
                          .groupby('hour')
                          .size()
                          .reindex(hourly_range, fill_value=0)
                          .rename('salidas'))

        df_hist = pd.DataFrame({
            'interval_start': hourly_range,
            'entradas': entries_per_hour.values,
            'salidas': exits_per_hour.values
        })

        df_hist['ocupacion_actual'] = df_hist['entradas'].cumsum() - df_hist['salidas'].cumsum()
        df_hist['ocupacion_actual'] = df_hist['ocupacion_actual'].clip(lower=0)

        df_hist['hora_del_dia'] = df_hist['interval_start'].dt.hour
        df_hist['dia_semana'] = df_hist['interval_start'].dt.weekday

        # Calculate 'ocupacion_futura' without dropping NaNs immediately
        df_hist['ocupacion_futura'] = df_hist['ocupacion_actual'].shift(-PREDICT_AHEAD_HOURS)
        # Calculate 'ocupacion_lag_1h' before dropping NaNs
        df_hist['ocupacion_lag_1h'] = df_hist['ocupacion_actual'].shift(1)

        # Get total active spaces
        df_spaces = self.db_manager.get_spaces(self.parking_id)
        total_active_spaces = int((df_spaces['is_active'] == 1).sum())
        
        if total_active_spaces == 0:
            logging.warning(f"Parking ID {self.parking_id}: No active spaces found.")
            return pd.DataFrame(), 0

        df_hist['ocup_frac_actual'] = df_hist['ocupacion_actual'] / total_active_spaces
        df_hist['ocup_frac_futura'] = df_hist['ocupacion_futura'] / total_active_spaces
        df_hist['ocup_frac_lag1h'] = df_hist['ocupacion_lag_1h'] / total_active_spaces

        # Now drop rows where 'ocupacion_futura' is NaN (last N rows)
        # And fill NaNs for 'ocup_frac_lag1h' at the beginning
        df_hist = df_hist.dropna(subset=['ocup_frac_futura']).reset_index(drop=True)
        df_hist['ocup_frac_lag1h'].fillna(method='bfill', inplace=True) # Fill initial NaNs

        df_features = df_hist[[
            'interval_start',
            'ocup_frac_actual',
            'ocup_frac_lag1h',
            'hora_del_dia',
            'dia_semana',
            'ocup_frac_futura'
        ]].copy()
        
        if len(df_features) < 2:
            logging.warning(f"Parking ID {self.parking_id}: Not enough data points ({len(df_features)}) to build features for training.")
            return pd.DataFrame(), 0

        logging.info(f"Parking ID {self.parking_id}: Historical dataset built with {len(df_features)} samples.")
        return df_features, total_active_spaces

    def train_model(self):
        """
        Trains a RandomForestRegressor model using historical data for the specific parking lot.
        Stores the trained model and total spaces in memory.
        """
        logging.info(f"Parking ID {self.parking_id}: Attempting to train model.")
        df_feat, total_spaces = self._build_historical_dataset()

        if df_feat.empty or total_spaces == 0:
            logging.warning(f"Parking ID {self.parking_id}: Skipping model training due to insufficient data or no active spaces.")
            self.model = None
            self.total_parking_spaces = 0
            return

        y = df_feat['ocup_frac_futura']
        X = df_feat[['ocup_frac_actual', 'ocup_frac_lag1h', 'hora_del_dia', 'dia_semana']]

        if len(X) < 2:
            logging.warning(f"Parking ID {self.parking_id}: Not enough data for train/test split ({len(X)} samples). Training on all available data.")
            X_train, y_train = X, y
            X_test, y_test = pd.DataFrame(), pd.Series()
        else:
            test_size_actual = 0.2 if len(X) * 0.2 >= 1 else 1.0 / len(X) # Ensure at least 1 sample in test if possible
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_actual, shuffle=False)

        rf = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)

        if not X_test.empty:
            preds_test = rf.predict(X_test)
            mae = mean_absolute_error(y_test, preds_test)
            logging.info(f"Parking ID {self.parking_id}: Model trained. MAE (fraction): {mae:.4f}")
        else:
            logging.info(f"Parking ID {self.parking_id}: Model trained (no test set due to limited data).")

        self.model = rf
        self.total_parking_spaces = total_spaces
        self.last_retrain_time = datetime.datetime.now()
        logging.info(f"Parking ID {self.parking_id}: Model training complete.")

    def predict_occupation(self) -> tuple[int, int, float] | None:
        """
        Predicts future occupation for the current parking lot.
        Returns (current_occupied_spaces, total_spaces, predicted_fraction).
        """
        if self.model is None or self.total_parking_spaces == 0:
            logging.info(f"Parking ID {self.parking_id}: Model not trained or no active spaces. Cannot predict.")
            return None

        df_spaces = self.db_manager.get_spaces(self.parking_id)
        current_occupied_spaces = int((df_spaces['is_occupied'] == 1).sum())
        total_active_spaces = int((df_spaces['is_active'] == 1).sum())
        
        if total_active_spaces == 0:
            logging.warning(f"Parking ID {self.parking_id}: No active spaces for prediction.")
            return None

        current_occupation_fraction = current_occupied_spaces / total_active_spaces
        
        # For lag1h: assume occupation 1 hour ago was similar to the last prediction,
        # or current if no previous prediction exists.
        lag1h_occupation_fraction = self.last_prediction_value if self.last_prediction_value is not None else current_occupation_fraction
        
        now = datetime.datetime.now()
        current_hour = now.hour
        day_of_week = now.weekday()

        # Prepare features as a DataFrame with explicit column names to avoid UserWarning
        X_new = pd.DataFrame([[
            current_occupation_fraction,
            lag1h_occupation_fraction,
            current_hour,
            day_of_week
        ]], columns=['ocup_frac_actual', 'ocup_frac_lag1h', 'hora_del_dia', 'dia_semana'])
        
        predicted_fraction = self.model.predict(X_new)[0]
        predicted_fraction = float(np.clip(predicted_fraction, 0.0, 1.0))

        self.last_prediction_value = predicted_fraction
        logging.info(f"Parking ID {self.parking_id}: Prediction made: Current {current_occupation_fraction:.2f}, Predicted {predicted_fraction:.2f}")
        return current_occupied_spaces, total_active_spaces, predicted_fraction

    def generate_alerts(self, current_occupied_spaces: int, total_spaces: int, predicted_fraction: float) -> list[dict]:
        """
        Generates alert messages based on current and predicted occupation thresholds.
        Returns a list of dictionaries, each with 'type' (e.g., 'warning', 'critical') and 'message'.
        """
        self.current_alerts.clear()

        absolute_threshold_critical = OCCUPANCY_ALERT_THRESHOLD * total_spaces
        absolute_threshold_warning = 0.80 * total_spaces # New warning threshold

        # Alert 1: Critical - Current occupation very high
        if current_occupied_spaces >= absolute_threshold_critical:
            msg = (f"Ocupación actual CRÍTICA ({current_occupied_spaces}/{total_spaces} "
                   f"o {current_occupied_spaces/total_spaces:.0%}).")
            self.current_alerts.append({'type': 'critical', 'message': msg})
            logging.critical(f"Parking ID {self.parking_id}: {msg}")
        # Alert 2: Warning - Current occupation approaching critical
        elif current_occupied_spaces >= absolute_threshold_warning:
            msg = (f"Ocupación actual ALTA ({current_occupied_spaces}/{total_spaces} "
                   f"o {current_occupied_spaces/total_spaces:.0%}).")
            self.current_alerts.append({'type': 'warning', 'message': msg})
            logging.warning(f"Parking ID {self.parking_id}: {msg}")

        # Alert 3: Critical - Prediction for 1 hour ahead indicates high occupation
        predicted_absolute = predicted_fraction * total_spaces
        if predicted_absolute >= absolute_threshold_critical:
            msg = (f"PREDICCIÓN CRÍTICA: En {PREDICT_AHEAD_HOURS} h, se espera alta ocupación "
                   f"({int(predicted_absolute)}/{total_spaces} o {predicted_fraction:.0%}).")
            self.current_alerts.append({'type': 'critical', 'message': msg})
            logging.critical(f"Parking ID {self.parking_id}: {msg}")
        # Alert 4: Warning - Prediction approaching critical
        elif predicted_absolute >= absolute_threshold_warning:
            msg = (f"PREDICCIÓN ALTA: En {PREDICT_AHEAD_HOURS} h, se espera alta ocupación "
                   f"({int(predicted_absolute)}/{total_spaces} o {predicted_fraction:.0%}).")
            self.current_alerts.append({'type': 'warning', 'message': msg})
            logging.warning(f"Parking ID {self.parking_id}: {msg}")

        # Alert 5: Warning - Significant change in prediction vs. current (>15% difference)
        diff = abs(predicted_fraction - (current_occupied_spaces / total_spaces))
        if diff >= 0.15: # Increased threshold for "brusque" change
            msg_type = 'warning'
            if diff >= 0.25: # Even higher difference is critical
                msg_type = 'critical'
            msg = (f"CAMBIO BRUSCO: Diferencia entre ocupación actual y predicción. "
                   f"Actual {current_occupied_spaces/total_spaces:.0%}, "
                   f"Predicción {predicted_fraction:.0%}.")
            self.current_alerts.append({'type': msg_type, 'message': msg})
            logging.log(logging.WARNING if msg_type == 'warning' else logging.CRITICAL,
                        f"Parking ID {self.parking_id}: {msg}")

        return list(self.current_alerts)

    def run_periodic_tasks(self):
        """
        Function to run in a separate thread for periodic tasks:
        1) Retrains the model initially and every RETRAIN_INTERVAL_SECONDS.
        2) Makes predictions and generates alerts if a model is available.
        3) Puts the data (current_occupied, total_spaces, predicted_fraction, alerts) into the queue for the GUI.
        4) Sleeps for POLL_INTERVAL_SECONDS and repeats.
        """
        logging.info(f"Parking ID {self.parking_id}: Starting periodic tasks.")
        
        # Initial model training
        try:
            self.train_model()
        except Exception as e:
            logging.error(f"Parking ID {self.parking_id}: Error training initial model: {e}")
            # If initial training fails, provide a fallback payload for the GUI
            payload = {
                'current_occupied': 0,
                'total_spaces': self.total_parking_spaces,
                'predicted_fraction': 0.0,
                'alerts': [{'type': 'error', 'message': f"ERROR: No se pudo entrenar el modelo ({e})."}],
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            if not self.data_queue.full():
                self.data_queue.put(payload)
            return # Exit thread if initial model fails to prevent constant errors

        next_retrain_time = time.time() + RETRAIN_INTERVAL_SECONDS

        while True:
            # Check if it's time to retrain the model
            if time.time() >= next_retrain_time:
                try:
                    self.train_model()
                except Exception as e:
                    logging.error(f"Parking ID {self.parking_id}: Error retraining model: {e}")
                next_retrain_time = time.time() + RETRAIN_INTERVAL_SECONDS

            # If model is available, predict and generate alerts
            try:
                res = self.predict_occupation()
                if res is not None:
                    current_occupied, total_spaces, predicted_fraction = res
                    alerts = self.generate_alerts(current_occupied, total_spaces, predicted_fraction)
                    payload = {
                        'current_occupied': current_occupied,
                        'total_spaces': total_spaces,
                        'predicted_fraction': predicted_fraction,
                        'alerts': alerts,
                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    if self.data_queue.full():
                        try:
                            self.data_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.data_queue.put(payload)
                else:
                    # If prediction failed (e.g., total_active_spaces is 0), send an appropriate message
                    payload = {
                        'current_occupied': 0,
                        'total_spaces': self.total_parking_spaces,
                        'predicted_fraction': 0.0,
                        'alerts': [{'type': 'info', 'message': "ADVERTENCIA: Datos no disponibles o espacios no activos."}],
                        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    if not self.data_queue.full():
                        self.data_queue.put(payload)

            except Exception as e:
                logging.error(f"Parking ID {self.parking_id}: Error in prediction or alerts: {e}", exc_info=True)
                payload = {
                    'current_occupied': 0,
                    'total_spaces': self.total_parking_spaces,
                    'predicted_fraction': 0.0,
                    'alerts': [{'type': 'error', 'message': f"ERROR: Fallo al actualizar información ({e})."}],
                    'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                if self.data_queue.full():
                    try:
                        self.data_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.data_queue.put(payload)

            time.sleep(POLL_INTERVAL_SECONDS)

    def get_latest_data(self):
        """Retrieves the latest data payload from the agent's queue."""
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

    def close_db_connection(self):
        """Closes the agent's database connection."""
        self.db_manager._close()