import numpy as np
import pandas as pd
from generator import RandomGenerator
from tqdm import tqdm


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
content_D = pd.read_csv(input_D, sep=',', index_col=0)
content_T = pd.read_csv(input_T)
nodes_df = pd.read_csv(nodes, sep=';', index_col='name_T')
content_P = pd.read_csv(input_P, sep=';', index_col=0).transpose().apply(pd.to_numeric)

# DataFrames parsing
D = parse_D(content_D, content_T)
T = parse_T(content_T, nodes_df)
P = parse_P(content_P, content_T)

# Matrices and vectors creation
mat_D = D.to_numpy()

vec_T = T.to_numpy()
vec_T = np.reshape(vec_T, (vec_T.shape[0],))

mat_P = P.to_numpy()

# Tuples testing
tuples = RandomGenerator(N_TUPLES, TUPLES_SIZE)

N = mat_D.shape[0] * mat_D.shape[1]

for t in tqdm(tuples, desc='Testing tuples', total=N_TUPLES):
    a1, a2, a3 = t
    w_D = np.multiply(a1, mat_D)
    w_T = np.multiply(a2, vec_T)

    S = w_D + w_T + a3

    O_R = np.sqrt((S - mat_P) ** 2) / N
