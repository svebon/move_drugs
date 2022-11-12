import numpy as np
import pandas as pd
import yaml
from data_parsing import *
from optimizers import *
import random
from math import ceil
from tabulate import tabulate
from time import time


def compare_optimizers(optimizers: list):
    """Compare optimizers on the same dataset and print results

    Parameters
    ----------
    optimizers : list
        List of optimizers to compare

    Returns
    -------
    None
    """
    [optimizer.optimize() for optimizer in optimizers]
    [print(f'{optimizer.__class__.__name__}: {optimizer.best_result.O_R_avg} | {optimizer.best_result.alphas}') for
     optimizer in optimizers]


def split_data(data: pd.DataFrame, train_recs: list, test_recs: list) -> tuple:
    """Split data into train and test sets.

    Split data into train and test sets, based on the receptors in the train and test sets.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe to split
    train_recs : list
        List of receptors in the train set
    test_recs : list
        List of receptors in the test set

    Returns
    -------
    tuple
        Tuple of train and test dataframes
    """
    train_data = data[train_recs]
    test_data = data[test_recs]
    return train_data, test_data


def split_receptors(receptors: list, train_ratio: float = 0.75) -> tuple:
    """Split receptors into train and test sets.

    Split receptors into train and test sets, based on the given ratio (train set size/dataset size).

    Parameters
    ----------
    receptors : list
        List of receptors to split
    train_ratio : float
        Ratio of train set size/dataset size

    Returns
    -------
    tuple
        Tuple of train and test receptors lists
    """
    train_size = ceil(len(receptors) * train_ratio)
    train = receptors[:train_size]
    test = receptors[train_size:]

    return train, test


def label_O_R(matrix: np.array, receptors: list, drugs: list) -> pd.DataFrame:
    """Label the O_R matrix with receptors and drugs names.

    Parameters
    ----------
    matrix : np.array
        O_R matrix
    receptors : list
        List of receptors
    drugs : list
        List of drugs

    Returns
    -------
    pd.DataFrame
        Dataframe with labeled O_R matrix
    """
    labeled_O_R = pd.DataFrame(matrix, index=drugs, columns=receptors)
    return labeled_O_R


def pick_random_drugs(drugs: list, n: int) -> list:
    """Pick n random drugs from the given list.

    Parameters
    ----------
    drugs : list
        List of drugs to pick from
    n : int
        Number of drugs to pick

    Returns
    -------
    list
        List of n random drugs
    """
    return random.sample(drugs, n)


def write_to_file():
    """Write results to file.

    Write the following information to file:
    - Train dataframes
    - Test dataframes
    - Optimizers results
    """
    file = open(f'data_{time()}.txt', 'w')

    file.write('Train D\n')
    file.write(str(train_D))

    file.write('\n\nTrain T\n')
    file.write(str(train_T))

    file.write('\n\nTrain P\n')
    file.write(str(train_P))

    file.write('\n\nTest D\n')
    file.write(str(test_D))

    file.write('\n\nTest T\n')
    file.write(str(test_T))

    file.write('\n\nTest P\n')
    file.write(str(test_P))

    file.write(f'\n\n{tabulate(results, headers=["Alphas", "Optimizer", "O_R_avg"], tablefmt="grid")}\n\n')
    file.close()


def drug_receptor_prob(drug: str, receptor: str) -> float:
    """Calculate the probability of a drug-receptor interaction

    Parameters
    ----------
    drug : str
        Drug name
    receptor : str
        Receptor name

    Returns
    -------
    float
        Probability of drug-receptor interaction
    """
    O_R_avgs = [O_R for tuple, opt_name, O_R in results]
    best_O_R_avg = min(O_R_avgs)
    best_index = O_R_avgs.index(best_O_R_avg)
    best_tuple, best_opt_name, best_O_R_avg = results[best_index]
    best_O_R = support_optimizer.get_O_R(best_tuple)
    best_dataset = label_O_R(best_O_R, test_recs, common_drugs)
    interaction_prob = best_dataset[receptor][drug]

    return interaction_prob


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

common_drugs = common_drugs(D.drugs, P.drugs)
# common_drugs = pick_random_drugs(common_drugs, 5)

D.filter_drugs(common_drugs)
P.filter_drugs(common_drugs)

# Data splitting
random.shuffle(common_recs)

train_recs, test_recs = split_receptors(common_recs)

train_D, test_D = split_data(D.data, train_recs, test_recs)
train_T, test_T = split_data(T.data, train_recs, test_recs)
train_P, test_P = split_data(P.data, train_recs, test_recs)

rand_optimizer = RandomOptimizer(train_D.to_numpy(), train_T.to_numpy(), train_P.to_numpy())
gp_optimizer = GPOptimizer(train_D.to_numpy(), train_T.to_numpy(), train_P.to_numpy(), min_imp_timeout=50)
bh_optimizer = BHOptimizer(train_D.to_numpy(), train_T.to_numpy(), train_P.to_numpy(), min_imp_timeout=10,
                           guess=[1, 1, 1])

optimizers = [rand_optimizer, gp_optimizer, bh_optimizer]

compare_optimizers(optimizers)

# Test phase

support_optimizer = Optimizer(test_D.to_numpy(), test_T.to_numpy(), test_P.to_numpy())

results = []
for optimizer in optimizers:
    best_tuple = optimizer.best_result.alphas
    O_R_avg = support_optimizer.get_O_R_avg(best_tuple)
    results.append([best_tuple, optimizer.__class__.__name__, O_R_avg])

print(tabulate(results, headers=['Alphas', 'Optimizer', 'O_R_avg'], tablefmt='grid'))

interaction_prob = drug_receptor_prob('<drug>', '<receptor>')

print(f'Interaction probability: {interaction_prob}')
