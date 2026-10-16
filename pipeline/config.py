from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'  # cleaned outputs of notebook/00_Data_Preparation.ipynb
RAW_DIR = ROOT / 'dwc26' / 'TestCases'  # organiser's starter kit
INTRO_PATH = RAW_DIR / '01 Introduction.xlsx'  # uncertainty parameters of the 100 cases
TEMPLATE_PATH = RAW_DIR / '09 Template Deliverable.xlsx'  # official copy: all 10 sheets run to 2028-01-01
OUT_DIR = ROOT / 'outputs'
FIG_DIR = OUT_DIR / 'figures' / 'pipeline'  # key charts from python -m pipeline.main (notebooks have their own folders)
RESULT_DIR = OUT_DIR / 'result'  # Excel outputs
DELIVERABLE_PATH = RESULT_DIR / '09 Template Deliverable.xlsx'
RESULTS_PATH = RESULT_DIR / 'history_match_results.xlsx'

SEED = 42
VARS = ['oil', 'gas', 'water']
UNITS = ['STB', 'MSCF', 'STB']
PARAM_COLS = ['Fault Transmissibility', 'Porosity Multiplier', 'Permeability Multiplier', 'Aquifer Pore Volume']

TRAIN_CASES = list(range(1, 71))
VAL_CASES = list(range(71, 86))
TEST_CASES = list(range(86, 101))

FC_DATES = pd.date_range('2008-04-01', '2028-01-01', freq='QS-JAN')  # 80 quarters, matches the template
DECLINE_WINDOW = 5  # quarters; chosen on validation backtests in notebook/01_Proxy_Model.ipynb
N_DELIVER = 10  # "pick the best 10 forecast cases"

# Chart styling shared with notebook/01_Proxy_Model.ipynb (one fixed colour per entity)
COLORS = {'actual': '#0b0b0b', 'matched': '#2a78d6', 'alt': '#eb6834', 'provided': '#c3c2b7', 'muted': '#898781'}
