from flask import Flask, render_template, jsonify, request
import threading
import time
import datetime
import atexit
import pandas as pd # Importar pandas aquí, ya que se usa en get_historical_occupation
import logging # Usar logging

from agent import AgentPark, POLL_INTERVAL_SECONDS
from database import DatabaseManager, DB_CONFIG

app = Flask(__name__)

# Configuración de logging para la aplicación Flask
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# Dictionary to hold active AgentPark instances, keyed by parking_id
active_agents: dict[int, AgentPark] = {}
# Global DB manager for fetching parking list
global_db_manager = DatabaseManager(DB_CONFIG)

def start_agent_for_parking(parking_id: int):
    """Starts an AgentPark instance in a new thread for a given parking ID."""
    if parking_id not in active_agents:
        logging.info(f"App: Starting agent for parking ID: {parking_id}")
        agent = AgentPark(parking_id)
        active_agents[parking_id] = agent
        # Start the agent's periodic tasks in a daemon thread
        thread = threading.Thread(target=agent.run_periodic_tasks, daemon=True)
        thread.start()
    else:
        logging.info(f"App: Agent already running for parking ID: {parking_id}")

@app.route('/')
def index():
    """Renders the main page, displaying a list of parking lots with basic info."""
    parkings_df = global_db_manager.get_parkings()
    parkings_data = []

    for _, parking in parkings_df.iterrows():
        parking_id = int(parking['id'])
        agent = active_agents.get(parking_id)
        current_data = None
        if agent:
            current_data = agent.get_latest_data() # Get latest data from agent's queue
        
        # Prepare data for display on index page
        if current_data:
            occupied_spaces = current_data.get('current_occupied', 0)
            total_spaces = current_data.get('total_spaces', 0)
            predicted_fraction = current_data.get('predicted_fraction', 0.0)
            status_text = "Cargando..."
            status_class = "status-info"

            if total_spaces > 0:
                current_occupancy_percentage = (occupied_spaces / total_spaces)
                predicted_occupancy_percentage = predicted_fraction

                if current_occupancy_percentage >= 0.90:
                    status_text = "CRÍTICO (Lleno)"
                    status_class = "status-critical"
                elif current_occupancy_percentage >= 0.80:
                    status_text = "ALTA (Casi lleno)"
                    status_class = "status-warning"
                else:
                    status_text = "NORMAL"
                    status_class = "status-ok"
            
            parkings_data.append({
                'id': parking_id,
                'name': parking['name'],
                'address': parking['address'],
                'occupied': occupied_spaces,
                'total': total_spaces,
                'predicted_fraction': predicted_fraction,
                'status_text': status_text,
                'status_class': status_class
            })
        else:
            # If no data yet from agent, or agent not started
            parkings_data.append({
                'id': parking_id,
                'name': parking['name'],
                'address': parking['address'],
                'occupied': "N/A",
                'total': "N/A",
                'predicted_fraction': 0.0,
                'status_text': "Iniciando Agente...",
                'status_class': "status-loading"
            })
        
        # Ensure agent is started for this parking if not already
        start_agent_for_parking(parking_id)

    return render_template('index.html', parkings=parkings_data)

@app.route('/parking/<int:parking_id>')
def parking_dashboard(parking_id: int):
    """
    Renders the dashboard for a specific parking lot.
    Initiates the agent for this parking if not already running.
    """
    start_agent_for_parking(parking_id)

    parking_info_df = global_db_manager.execute_query("SELECT name, address FROM parkings WHERE id = %s", params=(parking_id,))
    parking_name = parking_info_df['name'].iloc[0] if not parking_info_df.empty else f"Parqueadero {parking_id}"

    # Pass PREDICT_AHEAD_HOURS to template for dynamic display
    from agent import PREDICT_AHEAD_HOURS
    return render_template('dashboard.html', parking_id=parking_id, parking_name=parking_name, PREDICT_AHEAD_HOURS=PREDICT_AHEAD_HOURS)

@app.route('/api/data/<int:parking_id>')
def get_parking_data(parking_id: int):
    """
    API endpoint to provide real-time data for a specific parking lot.
    This is called by the frontend periodically.
    """
    agent = active_agents.get(parking_id)
    if agent:
        data = agent.get_latest_data()
        if data:
            return jsonify(data)
    
    # Return default or error data if agent is not running or no data
    return jsonify({
        'current_occupied': 0,
        'total_spaces': 0,
        'predicted_fraction': 0.0,
        'alerts': [{'type': 'info', 'message': "Esperando datos del agente..."}],
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route('/api/historical_occupation/<int:parking_id>')
def get_historical_occupation(parking_id: int):
    """
    API endpoint to provide historical occupation data for plotting.
    This fetches data directly from the DB for plotting.
    """
    # Create a new DBManager instance for this request to ensure thread safety
    # (though with proper SQLAlchemy connection pooling, this might be less necessary)
    db_manager = DatabaseManager(DB_CONFIG) 
    df_entries, df_exits = db_manager.get_entries_exits(parking_id)
    db_manager._close() # Close connection immediately after use

    # Ensure timestamps are datetime objects
    df_entries['timestamp'] = pd.to_datetime(df_entries['timestamp'], errors='coerce')
    df_exits['exit_ts'] = pd.to_datetime(df_exits['exit_ts'], errors='coerce')
    df_exits.dropna(subset=['exit_ts'], inplace=True) # Drop NaT exits

    # Calculate current occupied spaces based on entries without corresponding exits
    # This is a more accurate way to get 'current' occupation for plotting
    current_occupied = df_entries[~df_entries['id'].isin(df_exits['entry_id'])]
    initial_occupation = len(current_occupied)

    if df_entries.empty and df_exits.empty:
        logging.info(f"API: No historical data for plotting for parking ID {parking_id}.")
        return jsonify({'timestamps': [], 'occupations': []})

    # Get all unique timestamps from entries and exits, then create a continuous hourly range
    all_timestamps = pd.concat([df_entries['timestamp'], df_exits['exit_ts']]).dropna()
    if all_timestamps.empty:
         return jsonify({'timestamps': [], 'occupations': []})

    min_ts = all_timestamps.min().floor('h')
    max_ts = all_timestamps.max().ceil('h')
    
    # Limit to last 24 hours (or adjust as needed for chart)
    # Ensure min_hour does not go beyond max_hour
    start_plotting_from = datetime.datetime.now().replace(minute=0, second=0, microsecond=0) - datetime.timedelta(hours=23)
    if min_ts < start_plotting_from:
        min_ts = start_plotting_from
    
    # Ensure min_ts is not greater than max_ts
    if min_ts > max_ts:
        min_ts = max_ts - datetime.timedelta(hours=1) # Ensure at least one hour if possible

    hourly_range = pd.date_range(start=min_ts, end=max_ts, freq='H')
    if hourly_range.empty: # This can happen if only a single event in the 24h window
        return jsonify({'timestamps': [], 'occupations': []})

    # Calculate entries and exits per hour bin
    entries_count = df_entries.groupby(pd.Grouper(key='timestamp', freq='H')).size()
    exits_count = df_exits.groupby(pd.Grouper(key='exit_ts', freq='H')).size()

    # Reindex to our continuous hourly range, filling NaNs with 0
    entries_count = entries_count.reindex(hourly_range, fill_value=0)
    exits_count = exits_count.reindex(hourly_range, fill_value=0)

    # Calculate cumulative occupation
    # Start with initial_occupation at the beginning of the hourly_range
    # This needs careful calculation to be correct relative to the start of the plotting window
    
    # A more robust way: simulate occupation over time
    events = pd.concat([
        df_entries.assign(type='entry', time=df_entries['timestamp']),
        df_exits.assign(type='exit', time=df_exits['exit_ts'])
    ]).sort_values('time').reset_index(drop=True)

    occupations_over_time = []
    current_occup = initial_occupation # This should be the actual occupation at the *first* hour of hourly_range
                                       # For simplicity in this example, let's re-calculate more directly.
                                       # We need the occupation at the *start* of the 24-hour window.
                                       # This is complex without a full transaction log.
                                       # For now, we'll assume 0 at the start of the window and cumulate.
                                       # A better way would be to get the actual occupation at `start_plotting_from`.
    
    # Let's simplify: just plot the current occupation for the last 24 hours based on entries/exits within that window.
    # This won't be perfectly accurate if cars entered before the 24h window and are still there.
    # For a *true* historical graph, you'd need the occupation *at the beginning* of the history.
    # For now, let's use the net change in the last 24 hours and a starting point.
    
    # Fallback/simplified historical calculation for plotting:
    # Use the AgentPark's internal _build_historical_dataset for historical data as it's already doing this logic.
    # However, this method returns features for ML, not raw occupation. We need a dedicated history function.

    # Re-using logic from _build_historical_dataset to generate occupation for the chart
    df_hist_full, _ = AgentPark(parking_id)._build_historical_dataset() # Temporary agent to get history
    db_manager._connect() # Reconnect global manager if needed
    
    if df_hist_full.empty:
        return jsonify({'timestamps': [], 'occupations': []})

    # Limit to the actual last 24 hours for display
    cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=24)
    df_hist_display = df_hist_full[df_hist_full['interval_start'] >= cutoff_time].copy()
    
    # Convert fractional occupation back to absolute numbers for the graph
    agent_instance = active_agents.get(parking_id)
    total_spaces_for_chart = agent_instance.total_parking_spaces if agent_instance and agent_instance.total_parking_spaces > 0 else 1 # Avoid division by zero
    
    df_hist_display['occupation_absolute'] = (df_hist_display['ocup_frac_actual'] * total_spaces_for_chart).round().astype(int)

    timestamps = [ts.strftime("%H:%M") for ts in df_hist_display['interval_start']]
    occupations = df_hist_display['occupation_absolute'].tolist()

    return jsonify({'timestamps': timestamps, 'occupations': occupations})


@app.context_processor
def inject_poll_interval():
    """Makes POLL_INTERVAL_SECONDS available in Jinja2 templates."""
    return dict(POLL_INTERVAL_SECONDS=POLL_INTERVAL_SECONDS * 1000) # Convert to milliseconds for JS

@atexit.register
def cleanup_agents():
    """Ensures all database connections are closed when the Flask app exits."""
    logging.info("App: Shutting down agents and closing database connections...")
    for agent in active_agents.values():
        agent.close_db_connection()
    if global_db_manager.conn and global_db_manager.conn.is_connected():
        global_db_manager._close()
    logging.info("App: Cleanup complete.")

if __name__ == '__main__':
    # Initial connection for global_db_manager
    try:
        global_db_manager._connect()
        # Start agents for all parkings listed in the database on app startup
        parkings_on_startup = global_db_manager.get_parkings()
        for _, parking in parkings_on_startup.iterrows():
            start_agent_for_parking(int(parking['id']))

    except Exception as e:
        logging.critical(f"App: Failed to connect to database or start initial agents at startup: {e}")
        # Optionally exit or handle this more gracefully, e.g., by not starting the app
        # sys.exit(1)

    app.run(debug=True, use_reloader=False) # use_reloader=False because of threading issues with Flask's reloader