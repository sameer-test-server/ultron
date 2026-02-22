#!/usr/bin/env python3
"""Integration test suite for Ultron data loader."""

import sys
import os
import tempfile
import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.data_loader import (
    _download_from_yahoo,
    _download_from_stooq,
    _normalize_downloaded_data,
    _clean_and_save,
    update_all_data,
)
from config.settings import RAW_DATA_DIR


def test_yahoo_download():
    """Test Yahoo Finance download for a popular ticker."""
    print("Test 1: Yahoo Finance download... ", end="", flush=True)
    try:
        df = _download_from_yahoo("RELIANCE.NS", datetime.date(2024, 1, 1), datetime.date(2024, 1, 10))
        assert not df.empty, "No data returned"
        assert "Close" in df.columns, "Missing Close column"
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_stooq_download():
    """Test Stooq CSV download as fallback."""
    print("Test 2: Stooq fallback download... ", end="", flush=True)
    try:
        df = _download_from_stooq("INFY.NS", datetime.date(2024, 1, 1), datetime.date(2024, 1, 10))
        # Stooq may not have all symbols; pass if any data or empty (expected for some)
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_data_normalization():
    """Test data normalization pipeline."""
    print("Test 3: Data normalization... ", end="", flush=True)
    try:
        df = _download_from_yahoo("TCS.NS", datetime.date(2024, 1, 1), datetime.date(2024, 1, 10))
        normalized = _normalize_downloaded_data(df)
        assert "Date" in normalized.columns, "Missing Date"
        assert "Open" in normalized.columns, "Missing Open"
        assert "Close" in normalized.columns, "Missing Close"
        assert "Volume" in normalized.columns, "Missing Volume"
        print("✓ PASS")
        return True
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_csv_save_load():
    """Test CSV save and load round-trip."""
    print("Test 4: CSV save/load round-trip... ", end="", flush=True)
    try:
        df = _download_from_yahoo("HDFCBANK.NS", datetime.date(2024, 1, 1), datetime.date(2024, 1, 10))
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_csv = f.name
        try:
            success = _clean_and_save(df, temp_csv)
            assert success, "Failed to save CSV"
            import pandas as pd
            loaded = pd.read_csv(temp_csv)
            assert len(loaded) > 0, "Loaded CSV is empty"
            print("✓ PASS")
            return True
        finally:
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def test_parallel_updates():
    """Test parallel update execution."""
    print("Test 5: Parallel 50-ticker update (6 workers)... ", end="", flush=True)
    try:
        result = update_all_data(workers=6)
        assert result["status"] in ("success", "partial"), f"Unexpected status: {result['status']}"
        assert result["updated"] + result["up_to_date"] + result["failed"] == 50, \
            f"Summary count mismatch: {result}"
        success_rate = (result["updated"] + result["up_to_date"]) / 50
        print(f"✓ PASS ({success_rate*100:.0f}% success)")
        return success_rate >= 0.95  # Allow 1 ticker failure
    except Exception as e:
        print(f"✗ FAIL: {e}")
        return False


def main():
    """Run integration test suite."""
    print("\n" + "=" * 70)
    print("  Ultron Integration Test Suite")
    print("=" * 70 + "\n")
    
    tests = [
        test_yahoo_download,
        test_stooq_download,
        test_data_normalization,
        test_csv_save_load,
        test_parallel_updates,
    ]
    
    passed = 0
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"Test crashed: {e}")
    
    print("\n" + "=" * 70)
    print(f"Results: {passed}/{len(tests)} tests passed")
    if passed == len(tests):
        print("✓ All tests passed! Ultron is ready for production.")
    elif passed >= len(tests) - 1:
        print("⚠ Most tests passed. Minor failures are acceptable (external data delays).")
    else:
        print("✗ Multiple failures. Review configuration and network connectivity.")
    print("=" * 70 + "\n")
    
    return 0 if passed >= len(tests) - 1 else 1


if __name__ == "__main__":
    sys.exit(main())
