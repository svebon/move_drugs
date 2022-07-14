import numpy as np
import pandas as pd
from generator import RandomGenerator
from time import time

TUPLES_SIZE = 3
N_TUPLES = 10

input_D = 'C:/Users/Sveva/Desktop/ML_datasets/matrix_D.csv'
content_D = pd.read_csv(input_D, sep=',', index_col=0)

print(content_D['Drugs'])
content_D = content_D.drop(['Drugs'], axis=1)
content_D = content_D.apply(pd.to_numeric)

# print(content_D)
mat_D = content_D.to_numpy()
# print(mat_D)


input_T = 'C:/Users/Sveva/Desktop/ML_datasets/vector_T.csv'
content_T = pd.read_csv(input_T)

nodes = 'C:/Users/Sveva/Desktop/ML_datasets/new_mapped_name.csv'
nodes_df = pd.read_csv(nodes, sep=';', index_col='name_T')

content_T['mapped name'] = content_T["shared name"].apply(
    lambda x: nodes_df.loc[x]['name_D'] if x in nodes_df.index else None)

content_T = content_T[content_T['mapped name'].notna()]
content_T = content_T[['mapped name', 'BetweennessCentrality']]
content_T = content_T.set_index('mapped name', drop=True)

vec_T = content_T.to_numpy()

input_P = 'C:/Users/Sveva/Desktop/ML_datasets/matrix_P.csv'
content_P = pd.read_csv(input_P, sep=';', index_col=0).transpose().apply(pd.to_numeric)

#print(content_P.index)

# TEMPORANEO
content_D = content_D[[col for col in content_D.columns if col in content_T.index]]
mat_D = content_D.to_numpy()

content_P = content_P[[col for col in content_P.columns if col in content_T.index]]
mat_P = content_P.to_numpy()
# FINE TEMPORANEO

N = mat_D.shape[0] * mat_D.shape[1]

tuples = RandomGenerator(N_TUPLES, TUPLES_SIZE)

start = time()
for tupla in tuples:
    a1, a2, a3 = tupla
    # print(a1, a2, a3)

    w_D = np.multiply(a1, mat_D)
    w_T = np.multiply(a2, vec_T)
    w_T = np.reshape(w_T, (w_T.shape[0], ))  # il vettore viene create bi-dimensionale

    # mat_w_T = np.matrix([w_T for _ in range(len(content_D))])

    S = w_D + w_T + a3

    O_R = np.sqrt((S - mat_P) ** 2) / N

    #print(O_R)

end = time()

print((end - start) / N_TUPLES * 1*1000*1000)


