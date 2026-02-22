#!/usr/bin/env bash
# Load environment variables from .env file if present
if [ -f "$(dirname "$0")/../.env" ]; then
  export $(grep -v '^#' "$(dirname "$0")/../.env" | xargs)
fi

# Safe cron runner for Ultron: acquires an exclusive flock to avoid overlapping runs
BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$BASE_DIR/.venv/bin/python"
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"
LOCK_FILE="$LOG_DIR/ultron.lock"
LOG_FILE="$LOG_DIR/cron.log"

exec 200>"$LOCK_FILE"
flock -n 200 || exit 0

# Run the updater with parallelism and append to cron log
"$VENV" "$BASE_DIR/scripts/run_ultron.py" --parallel 6 >> "$LOG_FILE" 2>&1

# release lock by exiting (flock releases on fd close)
