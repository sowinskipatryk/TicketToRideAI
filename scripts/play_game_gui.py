"""GUI entry point for playing Ticket to Ride."""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from PyQt6.QtWidgets import QApplication
except ImportError:
    print("Error: PyQt6 is not installed.")
    print("Please install it with: pip install PyQt6")
    sys.exit(1)

from gui.main_window import MainWindow


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

