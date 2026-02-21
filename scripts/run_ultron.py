import datetime
import logging
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.data_loader import update_all_data


def _configure_logging():
    """Configure console + file logging for local and cron execution."""
    logs_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    log_file = os.path.join(logs_dir, "ultron.log")
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)

    # Keep logs focused on Ultron signals; third-party libraries can be noisy.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)


def main():
    _configure_logging()
    logger = logging.getLogger("ultron.runner")

    start_time = datetime.datetime.now(datetime.timezone.utc)
    logger.info("Run started at %s", start_time.isoformat())

    exit_code = 0
    try:
        summary = update_all_data()
        logger.info("Final summary: %s", summary.get("status", "unknown"))
    except Exception as error:
        exit_code = 1
        logger.error("Fatal error: %s", error)

    end_time = datetime.datetime.now(datetime.timezone.utc)
    logger.info("Run ended at %s", end_time.isoformat())
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
