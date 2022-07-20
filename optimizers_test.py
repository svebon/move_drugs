from optimizers import *
import pandas as pd
from data_parsing import *

# Data's paths
input_D = 'C:/Users/mirco/Projects/Sveva/ML_datasets/matrix_D.csv'
input_T = 'C:/Users/mirco/Projects/Sveva/ML_datasets/vector_T.csv'
input_P = 'C:/Users/mirco/Projects/Sveva/ML_datasets/matrix_P.csv'
nodes = 'C:/Users/mirco/Projects/Sveva/ML_datasets/new_mapped_name.csv'

# Data loading
D = DockingEnergies()

names_map = NamesMap()
names_map.load(nodes)

T = Topology(names_map)

P = Interactions()

D.load(input_D)
T.load(input_T)
P.load(input_P)

D.filter_receptors(T.receptors)
P.filter_receptors(T.receptors)

optimizer = RandomOptimizer(D.to_numpy(), T.to_numpy(), P.to_numpy())
result = optimizer.optimize()

print(f'Best tuple: {result["best_tuple"]}')
print(f'Best O_R: {result["best_O_R_avg"]}')

