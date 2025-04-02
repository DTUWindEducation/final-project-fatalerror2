import sys
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import src
project_root = Path(__file__).resolve().parents[1]
data_path = project_root / "inputs" / "Location1.csv"
data_dict = src.load_data(data_path)
src.plot_relativehumidity_2m(data_path)