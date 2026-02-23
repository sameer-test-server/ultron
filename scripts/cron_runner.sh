#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PY="$BASE_DIR/.venv/bin/python"
LOG_DIR="$BASE_DIR/logs"
LOCK_FILE="$LOG_DIR/ultron.lock"
LOCK_DIR_FALLBACK="$LOG_DIR/ultron.lockdir"
LOG_FILE="$LOG_DIR/cron.log"
BRANCH="main"
WORKERS="6"

mkdir -p "$LOG_DIR"

log() {
  printf '%s | %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" >> "$LOG_FILE"
}

load_env_file() {
  local env_file="$BASE_DIR/.env"
  if [ -f "$env_file" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$env_file"
    set +a
    log "Loaded environment from .env"
  fi
}

resolve_runtime_config() {
  BRANCH="${ULTRON_GIT_BRANCH:-main}"
  WORKERS="${ULTRON_PARALLEL_WORKERS:-6}"
}

git_repo_ready() {
  git -C "$BASE_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1
}

git_pull_latest() {
  if ! git_repo_ready; then
    log "Git repo not detected at $BASE_DIR, skipping pull."
    return 0
  fi

  if ! git -C "$BASE_DIR" diff --quiet || ! git -C "$BASE_DIR" diff --cached --quiet; then
    log "Local changes detected, skipping pre-run pull."
    return 0
  fi

  if git -C "$BASE_DIR" pull --ff-only origin "$BRANCH" >> "$LOG_FILE" 2>&1; then
    log "Git pull completed on $BRANCH"
  else
    log "Git pull failed on $BRANCH; continuing with local state."
  fi
}

run_ultron_update() {
  if [ ! -x "$VENV_PY" ]; then
    log "Python venv executable not found: $VENV_PY"
    return 1
  fi

  if "$VENV_PY" "$BASE_DIR/scripts/run_ultron.py" --parallel "$WORKERS" >> "$LOG_FILE" 2>&1; then
    log "Ultron update finished successfully."
    return 0
  fi

  log "Ultron update returned a non-zero exit code."
  return 1
}

git_commit_and_push_data() {
  if ! git_repo_ready; then
    log "Git repo not detected, skipping data commit/push."
    return 0
  fi

  local git_user_name
  local git_user_email
  git_user_name="$(git -C "$BASE_DIR" config --get user.name || true)"
  git_user_email="$(git -C "$BASE_DIR" config --get user.email || true)"

  # Optional self-configuration from .env for headless cron environments.
  if [ -z "$git_user_name" ] && [ -n "${ULTRON_GIT_USER_NAME:-}" ]; then
    git -C "$BASE_DIR" config user.name "$ULTRON_GIT_USER_NAME"
    git_user_name="$ULTRON_GIT_USER_NAME"
    log "Configured git user.name from ULTRON_GIT_USER_NAME"
  fi
  if [ -z "$git_user_email" ] && [ -n "${ULTRON_GIT_USER_EMAIL:-}" ]; then
    git -C "$BASE_DIR" config user.email "$ULTRON_GIT_USER_EMAIL"
    git_user_email="$ULTRON_GIT_USER_EMAIL"
    log "Configured git user.email from ULTRON_GIT_USER_EMAIL"
  fi

  if [ -z "$git_user_name" ] || [ -z "$git_user_email" ]; then
    # Last-resort local defaults to keep unattended cron commits working.
    git -C "$BASE_DIR" config user.name "${git_user_name:-Ultron Bot}" || true
    git -C "$BASE_DIR" config user.email "${git_user_email:-ultron-bot@local}" || true
    git_user_name="$(git -C "$BASE_DIR" config --get user.name || true)"
    git_user_email="$(git -C "$BASE_DIR" config --get user.email || true)"
    if [ -z "$git_user_name" ] || [ -z "$git_user_email" ]; then
      log "Git user.name/user.email not configured; skipping data commit."
      return 0
    fi
    log "Configured fallback local git identity for automation."
  fi

  # Stage only data CSV files so source code/manual edits are not auto-committed.
  git -C "$BASE_DIR" add -A -- data/raw >> "$LOG_FILE" 2>&1 || true

  if git -C "$BASE_DIR" diff --cached --quiet -- data/raw; then
    log "No CSV changes to commit."
    return 0
  fi

  local timestamp
  timestamp="$(date -u '+%Y-%m-%d %H:%M:%SZ')"
  local commit_msg="chore(data): daily csv sync ${timestamp}"

  if git -C "$BASE_DIR" commit --only --message "$commit_msg" -- data/raw >> "$LOG_FILE" 2>&1; then
    log "Committed CSV changes."
  else
    log "CSV commit failed; skipping push."
    return 1
  fi

  if git -C "$BASE_DIR" push origin "$BRANCH" >> "$LOG_FILE" 2>&1; then
    log "Pushed CSV commit to origin/$BRANCH"
  else
    log "Push failed for origin/$BRANCH"
    return 1
  fi
}

main() {
  if command -v flock >/dev/null 2>&1; then
    exec 200>"$LOCK_FILE"
    if ! flock -n 200; then
      log "Another cron run is active, exiting."
      exit 0
    fi
  else
    if ! mkdir "$LOCK_DIR_FALLBACK" 2>/dev/null; then
      log "Another cron run is active, exiting."
      exit 0
    fi
    trap 'rmdir "$LOCK_DIR_FALLBACK" 2>/dev/null || true' EXIT
  fi

  log "Cron run started."
  load_env_file
  resolve_runtime_config
  git_pull_latest

  if run_ultron_update; then
    git_commit_and_push_data || true
    log "Cron run finished."
    exit 0
  fi

  # Try to push partial CSV updates even when updater reports failure.
  git_commit_and_push_data || true
  log "Cron run finished with updater errors."
  exit 1
}

main "$@"
