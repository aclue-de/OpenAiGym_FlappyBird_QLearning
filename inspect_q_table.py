import numpy as np
from ast import literal_eval
from pandas import read_csv

q_table = read_csv("q_table.csv", names=["state_history", "actions"])

print(f"shape: {q_table.shape}")
