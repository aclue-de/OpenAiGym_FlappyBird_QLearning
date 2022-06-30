import numpy as np
from ast import literal_eval
from pandas import read_csv

q_table = read_csv("q_table.csv", names=["state_history", "actions"])
# state_index = q_table.index[q_table["state_history"].apply(
#     lambda x: x == print(x.split(",")))]
state_index = q_table.index[q_table["state_history"].apply(
    lambda x: str(x) == str([[44.0, -9.0], [43.0, -9.0], [43.0, -9.0], [42.0, -10.0]]))]

print(
    f"shape: {q_table.shape}")
print("state_index", state_index)
str = "[0.1, 0.2]"
print(type(str))
print(literal_eval(str)[0])
