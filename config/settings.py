import os

# Project root directory (ultron/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw OHLCV storage path
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

# Default historical backfill window for first-time downloads
HISTORY_YEARS = 5

# Ensure required local directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
