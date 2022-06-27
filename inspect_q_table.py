from pprint import pprint

import numpy as np

q_table = np.load("q_table_custom.npy")
# for actions in q_table:
#     if actions[0] is not 0.0 and actions[1] is not 0.0:
#         print(actions)
print(
    f"shape: {q_table.shape} | min: {np.min(q_table)} | mean: {np.mean(q_table)} | max: {np.max(q_table)}")
