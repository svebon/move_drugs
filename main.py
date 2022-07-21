import yaml
from data_parsing import *
from optimizers import *

TUPLES_SIZE = 3
N_TUPLES = 10

# Data's paths
with open('paths.yaml') as f:
    paths = yaml.load(f, Loader=yaml.SafeLoader)

# Data loading
names_map = NamesMap()
names_map.load(paths['nodes'])

D = DockingEnergies()
T = Topology(names_map)
P = Interactions()

D.load(paths['input_D'])
T.load(paths['input_T'])
P.load(paths['input_P'])

D.filter_receptors(T.receptors)
P.filter_receptors(T.receptors)

# optimizer = RandomOptimizer(D.to_numpy(), T.to_numpy(), P.to_numpy())
# optimizer = GPOptimizer(D.to_numpy(), T.to_numpy(), P.to_numpy(), min_imp_timeout=50)
optimizer = BHOptimizer(D.to_numpy(), T.to_numpy(), P.to_numpy(), min_imp_timeout=10, guess=[1, 1, 1])
result = optimizer.optimize()

print(f'Best tuple: {result["best_tuple"]}')
print(f'Best O_R: {result["best_O_R_avg"]}')
