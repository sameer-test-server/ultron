#!/usr/bin/env python3
"""Ultron data loader health check and status dashboard."""

import os
import sys
import pandas as pd
import datetime

# Add project to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config.settings import RAW_DATA_DIR
from config.nifty50 import NIFTY50_TICKERS


def check_data_freshness():
    """Report on data freshness and gaps."""
    print("\n📊 Ultron Data Freshness Report")
    print("=" * 70)
    
    today = datetime.date.today()
    fresh_count = 0  # Last row dated today
    current_count = 0  # Last row dated within 1-3 days
    stale_count = 0  # Last row > 3 days old
    missing_count = 0  # CSV doesn't exist
    
    missing_tickers = []
    stale_tickers = []
    
    for ticker in sorted(NIFTY50_TICKERS):
        csv_path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
        
        if not os.path.exists(csv_path):
            missing_tickers.append(ticker)
            missing_count += 1
            continue
        
        try:
            df = pd.read_csv(csv_path, parse_dates=["Date"])
            if df.empty:
                stale_tickers.append(ticker)
                stale_count += 1
                continue
            
            last_date = df["Date"].dropna().max()
            if pd.isna(last_date):
                stale_tickers.append(ticker)
                stale_count += 1
                continue
            
            days_old = (today - last_date.date()).days
            
            if days_old == 0:
                fresh_count += 1
                status = "✓ Fresh"
            elif days_old <= 3:
                current_count += 1
                status = f"⚠ Current ({days_old}d)"
            else:
                stale_tickers.append(ticker)
                stale_count += 1
                status = f"✗ Stale ({days_old}d)"
                
            print(f"  {status:20} {ticker:20} | rows={len(df):5} | last={last_date.date()}")
            
        except Exception as e:
            stale_tickers.append(ticker)
            stale_count += 1
            print(f"  ✗ Error          {ticker:20} | {str(e)[:40]}")
    
    print("\n" + "=" * 70)
    print(f"Summary:")
    print(f"  Fresh (today):        {fresh_count:3} / {len(NIFTY50_TICKERS)}")
    print(f"  Current (1-3d):       {current_count:3} / {len(NIFTY50_TICKERS)}")
    print(f"  Stale (>3d):          {stale_count:3} / {len(NIFTY50_TICKERS)}")
    print(f"  Missing (no CSV):     {missing_count:3} / {len(NIFTY50_TICKERS)}")
    
    overall_healthy = fresh_count + current_count >= len(NIFTY50_TICKERS) - 2
    status_emoji = "✓" if overall_healthy else "⚠"
    print(f"\n{status_emoji} Overall Status: {'HEALTHY' if overall_healthy else 'DEGRADED'}")
    
    if missing_tickers:
        print(f"\n⚠ Missing Tickers ({len(missing_tickers)}):")
        for t in missing_tickers:
            print(f"    - {t}")
    
    if stale_tickers:
        print(f"\n⚠ Stale Tickers ({len(stale_tickers)}):")
        for t in stale_tickers:
            print(f"    - {t}")
    
    return not (missing_count > 0 or stale_count > 5)  # Healthy if <5 stale


def check_logs():
    """Check recent log entries for errors."""
    print("\n📋 Recent Log Entries (last 10)")
    print("=" * 70)
    
    log_file = os.path.join(BASE_DIR, "logs", "ultron.log")
    if not os.path.exists(log_file):
        print("  No log file found yet.")
        return True
    
    try:
        with open(log_file, "r") as f:
            lines = f.readlines()[-10:]
        for line in lines:
            print(f"  {line.rstrip()}")
    except Exception as e:
        print(f"  Error reading logs: {e}")
        return False
    
    return True


def check_failed_tickers():
    """Check for today's failed ticker list."""
    print(f"\n🚨 Failed Tickers (if any)")
    print("=" * 70)
    
    logs_dir = os.path.join(BASE_DIR, "logs")
    today = datetime.date.today().isoformat()
    failed_file = os.path.join(logs_dir, f"failed_tickers_{today}.txt")
    
    if not os.path.exists(failed_file):
        print("  ✓ No failures recorded today.")
        return True
    
    try:
        with open(failed_file, "r") as f:
            failed = [line.strip() for line in f if line.strip()]
        if failed:
            print(f"  {len(failed)} tickers failed to update:")
            for ticker in failed:
                print(f"    - {ticker}")
            return False
    except Exception as e:
        print(f"  Error reading failed tickers: {e}")
        return False
    
    return True


def main():
    """Run all checks."""
    print("\n" + "=" * 70)
    print("  Ultron Health Check Dashboard")
    print("=" * 70)
    
    checks = [
        ("Data Freshness", check_data_freshness),
        ("Log Health", check_logs),
        ("Failed Tickers", check_failed_tickers),
    ]
    
    all_pass = True
    for name, check_func in checks:
        try:
            if not check_func():
                all_pass = False
        except Exception as e:
            print(f"\n❌ {name} check failed: {e}")
            all_pass = False
    
    print("\n" + "=" * 70)
    if all_pass:
        print("✓ All checks passed. Ultron is healthy!")
    else:
        print("⚠ Some checks failed. Review above for details.")
    print("=" * 70 + "\n")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
