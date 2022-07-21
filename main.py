from data_parsing import *
from optimizers import *

TUPLES_SIZE = 3
N_TUPLES = 10

# Data's paths
input_D = 'C:/Users/Sveva/Desktop/ML_datasets/matrix_D.csv'
input_T = 'C:/Users/Sveva/Desktop/ML_datasets/vector_T.csv'
input_P = 'C:/Users/Sveva/Desktop/ML_datasets/matrix_P.csv'
nodes = 'C:/Users/Sveva/Desktop/ML_datasets/new_mapped_name.csv'

# Data loading
names_map = NamesMap()
names_map.load(nodes)

D = DockingEnergies()
T = Topology(names_map)
P = Interactions()

D.load(input_D)
T.load(input_T)
P.load(input_P)

D.filter_receptors(T.receptors)
P.filter_receptors(T.receptors)

# optimizer = RandomOptimizer(D.to_numpy(), T.to_numpy(), P.to_numpy())
# optimizer = GPOptimizer(D.to_numpy(), T.to_numpy(), P.to_numpy(), min_imp_timeout=50)
optimizer = BHOptimizer(D.to_numpy(), T.to_numpy(), P.to_numpy(), min_imp_timeout=10, guess=[1, 1, 1])
result = optimizer.optimize()

print(f'Best tuple: {result["best_tuple"]}')
print(f'Best O_R: {result["best_O_R_avg"]}')
