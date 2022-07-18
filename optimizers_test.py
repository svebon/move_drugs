from optimizers import *
import pandas as pd
import main

# Data's paths
input_D = 'C:/Users/Sveva/Desktop/ML_datasets/matrix_D.csv'
input_T = 'C:/Users/Sveva/Desktop/ML_datasets/vector_T.csv'
input_P = 'C:/Users/Sveva/Desktop/ML_datasets/matrix_P.csv'
nodes = 'C:/Users/Sveva/Desktop/ML_datasets/new_mapped_name.csv'

# Data loading
content_D = pd.read_csv(input_D, sep=',', index_col=0)
content_T = pd.read_csv(input_T)
nodes_df = pd.read_csv(nodes, sep=';', index_col='name_T')
content_P = pd.read_csv(input_P, sep=';', index_col=0).transpose().apply(pd.to_numeric)

# DataFrames parsing
D = main.parse_D(content_D, content_T)
T = main.parse_T(content_T, nodes_df)
P = main.parse_P(content_P, content_T)

# Matrices and vectors creation
mat_D = D.to_numpy()

vec_T = T.to_numpy()
vec_T = np.reshape(vec_T, (vec_T.shape[0],))

mat_P = P.to_numpy()

optimizer = RandomOptimizer(mat_D, vec_T, mat_P)
t, O_R = optimizer.optimize()

print(f'Best tuple: {t}')
print(f'Best O_R: {O_R}')

