import argparse
import datetime
import logging
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core import data_loader
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


def _check_virtualenv():
    """Warn if the script is not running under the project's virtualenv."""
    venv_path = os.path.join(BASE_DIR, ".venv")
    if venv_path not in sys.executable:
        print("WARNING: It looks like you're not running inside the project's .venv.")
        print(f"Current python: {sys.executable}")
        print(f"Recommended: {os.path.join(venv_path, 'bin', 'python')}")


def main(argv: list[str] | None = None):
    _configure_logging()
    logger = logging.getLogger("ultron.runner")

    parser = argparse.ArgumentParser(description="Run Ultron data updater")
    parser.add_argument(
        "--tickers",
        help="Comma-separated list of tickers to update (default: all NIFTY50)",
        default=None,
    )
    parser.add_argument(
        "--parallel",
        type=int,
        help="Number of worker threads for parallel downloads (default: 1)",
        default=1,
    )
    args = parser.parse_args(argv)

    _check_virtualenv()

    # Allow limiting tickers for quicker runs by monkey-patching the loader's ticker list.
    if args.tickers:
        custom = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if custom:
            logger.info("Limiting run to tickers: %s", ",".join(custom))
            print(f"Limiting run to tickers: {','.join(custom)}")
            data_loader.NIFTY50_TICKERS = custom

    start_time = datetime.datetime.now(datetime.timezone.utc)
    logger.info("Run started at %s", start_time.isoformat())

    exit_code = 0
    try:
        workers = args.parallel if hasattr(args, "parallel") else None
        summary = update_all_data(workers=workers)
        logger.info("Final summary: %s", summary.get("status", "unknown"))
    except KeyboardInterrupt:
        exit_code = 2
        logger.warning("Run interrupted by user")
    except Exception as error:
        exit_code = 1
        logger.error("Fatal error: %s", error)

    end_time = datetime.datetime.now(datetime.timezone.utc)
    logger.info("Run ended at %s", end_time.isoformat())
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
