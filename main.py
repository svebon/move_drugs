import yaml
from data_parsing import *
from optimizers import *


def compare_optimizers(optimizers: list):
    [optimizer.optimize() for optimizer in optimizers]
    [print(f'{optimizer.__class__.__name__}: {optimizer.best_result.O_R_avg} | {optimizer.best_result.alphas}') for
     optimizer in optimizers]


# Data's paths
with open('paths.yaml') as f:
    paths = yaml.load(f, Loader=yaml.SafeLoader)

# Data loading
names_map = NamesMap()
names_map.load(paths['nodes'])

D = DockingEnergies(names_map)
T = Topology()
P = Interactions(names_map)

D.load(paths['input_D'])
T.load(paths['input_T'])
P.load(paths['input_P'])

common_recs = common_receptors(D.receptors, T.receptors, P.receptors)

D.filter_receptors(common_recs)
T.filter_receptors(common_recs)
P.filter_receptors(common_recs)

rand_optimizer = RandomOptimizer(D.to_numpy(), T.to_numpy(), P.to_numpy())
gp_optimizer = GPOptimizer(D.to_numpy(), T.to_numpy(), P.to_numpy(), min_imp_timeout=50)
bh_optimizer = BHOptimizer(D.to_numpy(), T.to_numpy(), P.to_numpy(), min_imp_timeout=10, guess=[1, 1, 1])

compare_optimizers([rand_optimizer, gp_optimizer, bh_optimizer])
