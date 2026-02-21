import sys
from config import settings
from core.data_loader import update_all_data

def main():
    """
    Ultron v1 Entry Point.
    Currently configured to run the Data Engine.
    """
    print("🤖 Ultron v1 - Quantitative Trading Engine Initializing...")
    print(f"📂 Data Directory: {settings.RAW_DATA_DIR}")
    
    # Run the data update process
    update_all_data()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Execution interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n🔥 Fatal System Error: {e}")
        sys.exit(1)
