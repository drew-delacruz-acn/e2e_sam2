from datetime import datetime

# Global debug log collector
DEBUG_LOGS = []

def debug_log(message: str):
    """Collect debug messages for saving to file."""
    print(message)  # Still print to console
    DEBUG_LOGS.append(f"{datetime.now().strftime('%H:%M:%S')} | {message}")

def get_debug_logs():
    """Get all collected debug logs."""
    return DEBUG_LOGS

def clear_debug_logs():
    """Clear the debug log collector."""
    global DEBUG_LOGS
    DEBUG_LOGS = [] 