import pandas as pd


def common_receptors(*args) -> list:
    intersection = set.intersection(*map(set, args))
    return list(intersection)


class NamesMap:
    def __init__(self):
        self.map = {}

    def load(self, path: str):
        with open(path, 'r') as f:
            f.readline()  # skip header
            lines = [l.strip() for l in f.readlines()]

        pairs = [l.split(';') for l in lines]

        for pair in pairs:
            if pair[0] not in self.map:
                self.map[pair[0]] = pair[1]

            if pair[1] not in self.map:
                self.map[pair[1]] = pair[0]

    def map_name(self, name: str) -> str:
        return self.map[name] if name in self.map else None


class CustomDataFrame:
    def __init__(self):
        self.data = None
        self.csv_sep = ';'

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
        common_receptors = [rec for rec in self.receptors if rec in receptors]
        self.data = self.data[common_receptors]

    def to_numpy(self):
        if isinstance(self.data, pd.DataFrame):
            return self.data.to_numpy()

        return self.data.to_numpy().reshape(self.data.shape[0], )  # Series to 1D array


class DockingEnergies(CustomDataFrame):
    def __init__(self, names_map: NamesMap):
        super().__init__()
        self.csv_sep = ','
        self.names_map = names_map

    def load(self, path: str):
        super().load(path)
        self.data = self.data.set_index('Drugs', drop=True)
        self.remove_unnamed_cols()
        self.data.columns = self.data.columns.map(self.names_map.map_name)
        self.sort()
        self.data = self.data.apply(pd.to_numeric)

    @property
    def receptors(self):
        return self.data.columns.tolist()


class Interactions(CustomDataFrame):
    def __init__(self, names_map: NamesMap):
        super().__init__()
        self.names_map = names_map

    def load(self, path: str):
        super().load(path)
        self.data = self.data.set_index(self.data.columns[0], drop=True)
        self.data.index = self.data.index.rename('Receptors')
        self.data = self.data.transpose()
        self.remove_unnamed_cols()
        self.data.columns = self.data.columns.map(self.names_map.map_name)
        self.sort()
        self.data = self.data.apply(pd.to_numeric)
        self.data.index = self.data.index.rename('Drugs')

    @property
    def receptors(self):
        return self.data.columns.tolist()


class Topology(CustomDataFrame):
    def __init__(self):
        super().__init__()
        self.csv_sep = ','

    def load(self, path: str):
        super().load(path)
        self.data = self.data.set_index(self.data.columns[0], drop=True)
        self.data = self.data.set_index('shared name', drop=True)
        self.sort()
        self.data = self.data['BetweennessCentrality']

    @property
    def receptors(self):
        return self.data.index.tolist()


if __name__ == '__main__':
    name_map = NamesMap()
    name_map.load('C:/Users/mirco/Projects/Sveva/ML_datasets/rec_name_map.csv')

    D = DockingEnergies(name_map)
    D.load('C:/Users/mirco/Projects/Sveva/ML_datasets/matrix_D.csv')
    print(D.data)

    P = Interactions(name_map)
    P.load('C:/Users/mirco/Projects/Sveva/ML_datasets/matrix_P.csv')
    print(P.data)

    T = Topology()
    T.load('C:/Users/mirco/Projects/Sveva/ML_datasets/vector_T.csv')
    print(T.data)

    print(D.data[T.receptors])

    print(P.data[T.receptors])
