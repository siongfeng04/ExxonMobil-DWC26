from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'  # prepared inputs from notebook/DWC2026.ipynb (committed to the repo)
RAW_DIR = ROOT / 'dwc26' / 'TestCases'
TEMPLATE_PATH = RAW_DIR / '09 Template Deliverable.xlsx'  # official copy: all 10 sheets run to 2028-01-01
OUT_DIR = ROOT / 'outputs'
FIG_DIR = OUT_DIR / 'figures'
DELIVERABLE_PATH = OUT_DIR / '09 Template Deliverable.xlsx'

SEED = 42
VARS = ['oil', 'gas', 'water']
UNITS = ['STB', 'MSCF', 'STB']
PARAM_COLS = ['Fault Transmissibility', 'Porosity Multiplier', 'Permeability Multiplier', 'Aquifer Pore Volume']

TRAIN_CASES = list(range(1, 71))
VAL_CASES = list(range(71, 86))
TEST_CASES = list(range(86, 101))

FC_DATES = pd.date_range('2008-04-01', '2028-01-01', freq='QS-JAN')  # 80 quarters, matches the template
DECLINE_WINDOW = 5  # quarters; chosen on validation backtests in notebook/ML_Proxy_Model.ipynb
N_DELIVER = 10  # "pick the best 10 forecast cases"

# Chart styling shared with notebook/ML_Proxy_Model.ipynb (one fixed colour per entity)
COLORS = {'actual': '#0b0b0b', 'matched': '#2a78d6', 'alt': '#eb6834', 'provided': '#c3c2b7', 'muted': '#898781'}
