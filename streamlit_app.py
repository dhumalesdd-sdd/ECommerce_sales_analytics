"""
Streamlit Community Cloud Entry Point
Executes the main application logic from app.py.
"""
import runpy
from pathlib import Path

app_path = Path(__file__).parent / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
