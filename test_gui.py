import sys
from PySide6.QtWidgets import QApplication, QLabel, QWidget

# --- Check for key libraries ---
try:
    import numpy
    import pyarrow
    print("✅ NumPy and PyArrow are available.")
except ImportError as e:
    print(f"❌ Error importing a library: {e}")

# --- Test Qt GUI ---
print("🚀 Attempting to launch Qt window...")
app = QApplication(sys.argv)
window = QWidget()
window.setWindowTitle("Setup Test Successful!")
label = QLabel("If you can see this window, your Qt setup is working correctly.", parent=window)
window.resize(400, 100)
window.show()

sys.exit(app.exec())