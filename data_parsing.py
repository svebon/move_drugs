import pandas as pd
from typing import Union


class CustomDataFrame:
    data: Union[pd.DataFrame, pd.Series]
    csv_sep = ';'

    def sort(self):
        self.data = self.data.sort_index(axis='index')
        self.data = self.data.reindex(sorted(self.data.columns), axis=1)

    def remove_unnamed_cols(self):
        self.data = self.data.drop([col for col in self.data.columns if 'Unnamed' in col], axis='columns')

    def load(self, path: str):
        self.data = pd.read_csv(path, sep=self.csv_sep)

    @property
    def receptors(self):
        raise NotImplementedError

    def filter_receptors(self, receptors: list):
        self.data = self.data[receptors]

    def to_numpy(self):
        if isinstance(self.data, pd.DataFrame):
            return self.data.to_numpy()

        return self.data.to_numpy().reshape(self.data.shape[0], )  # Series to 1D array


class DockingEnergies(CustomDataFrame):
    def __init__(self):
        self.csv_sep = ','

    def load(self, path: str):
        super().load(path)
        self.data = self.data.set_index('Drugs', drop=True)
        self.remove_unnamed_cols()
        self.sort()
        self.data = self.data.apply(pd.to_numeric)

    @property
    def receptors(self):
        return self.data.columns.tolist()


class Interactions(CustomDataFrame):
    def load(self, path: str):
        super().load(path)
        self.data = self.data.set_index(self.data.columns[0], drop=True)
        self.data.index = self.data.index.rename('Receptors')
        self.data = self.data.transpose()
        self.remove_unnamed_cols()
        self.sort()
        self.data = self.data.apply(pd.to_numeric)
        self.data.index = self.data.index.rename('Drugs')

    @property
    def receptors(self):
        return self.data.columns.tolist()


class NamesMap:
    map: dict

    def load(self, path: str):
        with open(path, 'r') as f:
            f.readline()  # skip header
            lines = [l.strip() for l in f.readlines()]

        self.map = {l.split(';')[0]: l.split(';')[1] for l in lines}
        self.map = {l.split(';')[1]: l.split(';')[0] for l in lines}

    def map_name(self, name: str) -> str:
        return self.map[name] if name in self.map else None


class Topology(CustomDataFrame):
    def __init__(self, map: NamesMap):
        self.map = map
        self.csv_sep = ','

    def load(self, path: str):
        super().load(path)
        self.data = self.data.set_index(self.data.columns[0], drop=True)
        self.data['mapped_name'] = self.data['shared name'].apply(lambda x: self.map.map_name(x))
        self.data = self.data[self.data['mapped_name'].notna()]
        self.data = self.data.set_index('mapped_name', drop=True)
        self.sort()
        self.data = self.data['BetweennessCentrality']

    @property
    def receptors(self):
        return self.data.index.tolist()


if __name__ == '__main__':
    D = DockingEnergies()
    D.load('C:/Users/mirco/Projects/Sveva/ML_datasets/matrix_D.csv')
    print(D.data)

    P = Interactions()
    P.load('C:/Users/mirco/Projects/Sveva/ML_datasets/matrix_P.csv')
    print(P.data)

    name_map = NamesMap()
    name_map.load('C:/Users/mirco/Projects/Sveva/ML_datasets/rec_name_map.csv')

    T = Topology(name_map)
    T.load('C:/Users/mirco/Projects/Sveva/ML_datasets/vector_T.csv')
    print(T.data)

    print(D.data[T.receptors])

    print(P.data[T.receptors])
