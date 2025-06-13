import mysql.connector
import pandas as pd
import datetime
import logging # Usar logging

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': '',
    'database': 'autopark',
    'charset': 'utf8mb4'
}

class DatabaseManager:
    """
    Manages database connections and queries for the parking system.
    Encapsulates all DB operations to keep AgentPark cleaner.
    """
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None

    def _connect(self):
        """Establishes a connection to the MySQL database."""
        try:
            self.conn = mysql.connector.connect(**self.db_config)
            self.conn.autocommit = True  # Enable autocommit
            logging.info("DB: Database connected successfully.")
        except mysql.connector.Error as err:
            logging.error(f"DB: Error connecting to DB: {err}")
            raise

    def _close(self):
        """Closes the database connection."""
        if self.conn and self.conn.is_connected():
            self.conn.close()
            self.conn = None
            logging.info("DB: Database connection closed.")

    def execute_query(self, query: str, params=None) -> pd.DataFrame:
        """Executes a SELECT query and returns data as a Pandas DataFrame."""
        if not self.conn or not self.conn.is_connected():
            self._connect() # Reconnect if connection is lost
        try:
            # Pandas will give a UserWarning here if not using SQLAlchemy engine
            return pd.read_sql(query, con=self.conn, params=params)
        except mysql.connector.Error as err:
            logging.error(f"DB: Error executing query: {err}")
            raise
        except Exception as e:
            logging.error(f"DB: An unexpected error occurred during query: {e}")
            raise

    def get_parkings(self) -> pd.DataFrame:
        """Fetches all parking lots from the 'parkings' table."""
        query = "SELECT id, name, address FROM parkings"
        return self.execute_query(query)

    def get_spaces(self, parking_id: int) -> pd.DataFrame:
        """Fetches spaces for a specific parking ID."""
        query = "SELECT id, is_active, is_occupied, parking_id, vehicle_type_id FROM spaces WHERE parking_id = %s"
        return self.execute_query(query, params=(parking_id,))

    def get_entries_exits(self, parking_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetches entries and exits for a specific parking ID.
        Joins entries with exits and filters by parking_id via space_id.
        """
        df_entries = self.execute_query(
            "SELECT e.id, e.timestamp, e.space_id FROM entries e "
            "JOIN spaces s ON e.space_id = s.id WHERE s.parking_id = %s",
            params=(parking_id,)
        )
        df_exits = self.execute_query(
            "SELECT ex.id AS exit_id, ex.timestamp AS exit_ts, ex.amount_charged, "
            "ent.timestamp AS entry_ts, ent.space_id, ent.id as entry_id " # Added entry_id for join logic
            "FROM exits ex JOIN entries ent ON ex.entry_id = ent.id "
            "JOIN spaces s ON ent.space_id = s.id WHERE s.parking_id = %s",
            params=(parking_id,)
        )
        return df_entries, df_exits