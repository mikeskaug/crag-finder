import os

DATA_DIR = os.path.abspath("./data")
METADATA = os.path.join(DATA_DIR, 'training.csv')
SET_FRACTIONS = {'train': 0.8, 'validation': 0.1, 'test': 0.1}
LOG_DIR = os.path.abspath('./logs') 