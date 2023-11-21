import pandas as pd
import numpy as np

data = {
    'source': ['A', 'A', 'B', 'B', 'C', 'C'],
    'target': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
    'feature': [5, 3, 8, 2, 7, 1],
    'http_status': [200, 404, 200, 500, 200, 301]
}

input_df = pd.DataFrame(data)
input_df = input_df.set_index(keys=['source', 'target'], drop=True).sort_index()

source = 'A'
target = 'X'

empirical = np.sort(input_df.loc[(source, target), 'feature'].values)
print(empirical)
