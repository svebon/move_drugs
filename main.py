import numpy as np
import pandas as pd
from generator import RandomGenerator
from tqdm import tqdm
from data_parsing import *
from optimizers import *


def parse_D(D_df: pd.DataFrame, T_df: pd.DataFrame) -> pd.DataFrame:
    df = D_df.copy()
    df = df.reindex(sorted(df.columns), axis=1)
    df = df.drop(['Drugs'], axis=1)
    df = df.apply(pd.to_numeric)
    df = df[[col for col in df.columns if col in T_df.index]]
    return df


def parse_T(T_df: pd.DataFrame, nodes_df: pd.DataFrame) -> pd.DataFrame:
    df = T_df.copy()
    df = df.reindex(sorted(df.columns), axis=1)
    df['mapped name'] = df["shared name"].apply(lambda x: nodes_df.loc[x]['name_D'] if x in nodes_df.index else None)
    df = df[df['mapped name'].notna()]
    df = df[['mapped name', 'BetweennessCentrality']]
    df = df.set_index('mapped name', drop=True)
    return df


def parse_P(P_df: pd.DataFrame, T_df: pd.DataFrame) -> pd.DataFrame:
    df = P_df.copy()
    df = df.reindex(sorted(df.columns), axis=1)
    df = df[[col for col in df.columns if col in T_df.index]]
    return df


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
