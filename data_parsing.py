import pandas as pd


def common_receptors(*args) -> list:
    """Find common receptors in all dataframes

    Parameters
    ----------
    args : list
        List of dataframes

    Returns
    -------
    list
        List of common receptors
    """
    intersection = set.intersection(*map(set, args))
    return list(intersection)


def common_drugs(*args) -> list:
    """Find common drugs in all dataframes

    Parameters
    ----------
    args : list
        List of dataframes

    Returns
    -------
    list
        List of common drugs
    """
    intersection = set.intersection(*map(set, args))
    return list(intersection)


class NamesMap:
    """Map names in dataset

    Given a CSV file, with a pair of names in each row, create a symmetric map
    """

    def __init__(self):
        self.map = {}

    def load(self, path: str):
        """Load CSV file with name pairs

        Load CSV file with name pairs, and populate the map.

        Parameters
        ----------
        path : str
            Path to CSV file

        Returns
        -------
        None
        """
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
        """Map name

        Map a name to its equivalent in the map.

        Parameters
        ----------
        name : str
            Name to map

        Returns
        -------
        str
            Mapped name
        """
        return self.map[name] if name in self.map else None


class CustomDataFrame:
    """Custom DataFrame class that implements some useful methods"""

    def __init__(self):
        self.data = None
        self.csv_sep = ';'

    def sort(self):
        """Sort dataframe by index and columns"""

        self.data: pd.DataFrame
        self.data = self.data.sort_index(axis='index')
        self.data = self.data.reindex(sorted(self.data.columns), axis=1)

    def remove_unnamed_cols(self):
        """Remove unnamed columns"""

        self.data = self.data.drop([col for col in self.data.columns if 'Unnamed' in col], axis='columns')

    def remove_none_cols(self):
        """Remove columns without name"""

        self.data = self.data.drop(labels=None, axis='columns')

    def load(self, path: str):
        """Load CSV file into dataframe"""

        self.data = pd.read_csv(path, sep=self.csv_sep)

    @property
    def receptors(self):
        raise NotImplementedError

    @property
    def drugs(self):
        raise NotImplementedError

    def filter_receptors(self, receptors: list):
        """Filter dataframe by the given receptors"""
        common_receptors = [rec for rec in self.receptors if rec in receptors]
        self.data = self.data[common_receptors]

    def filter_drugs(self, drugs: list):
        """Filter dataframe by the given drugs"""
        common_drugs = [drug for drug in self.data.index if drug in drugs]
        self.data = self.data.loc[common_drugs]

    def to_numpy(self):
        """Convert dataframe to 1D numpy array"""
        if isinstance(self.data, pd.DataFrame):
            return self.data.to_numpy()

        return self.data.to_numpy().reshape(self.data.shape[0], )  # Series to 1D array


class DockingEnergies(CustomDataFrame):
    """Docking energies dataframe

    Subclass of CustomDataFrame where the index is the drugs and the columns are the receptors
    """
    def __init__(self, names_map: NamesMap):
        super().__init__()
        self.csv_sep = ','
        self.names_map = names_map

    @property
    def drugs(self):
        return self.data.index.tolist()

    def load(self, path: str):
        super().load(path)
        self.data = self.data.set_index('Drugs', drop=True)
        self.remove_unnamed_cols()
        self.data.columns = self.data.columns.map(self.names_map.map_name)
        # self.remove_none_cols()
        self.sort()
        self.data = self.data.apply(pd.to_numeric)

    @property
    def receptors(self):
        return self.data.columns.tolist()


class Interactions(CustomDataFrame):
    """Interactions dataframe

    Subclass of CustomDataFrame where the index is the drugs and the columns are the receptors
    """
    def __init__(self, names_map: NamesMap):
        super().__init__()
        self.csv_sep = ','
        self.names_map = names_map

    def load(self, path: str):
        super().load(path)
        self.data = self.data.set_index(self.data.columns[0], drop=True)
        self.data.index = self.data.index.rename('Receptors')
        # self.data.columns = self.data.columns.map(self.names_map.map_name)
        # self.data = self.data.transpose()
        self.remove_unnamed_cols()
        # self.remove_none_cols()
        self.sort()
        self.data = self.data.apply(pd.to_numeric)
        self.data.index = self.data.index.rename('Drugs')

    @property
    def drugs(self):
        return self.data.index.tolist()

    @property
    def receptors(self):
        return self.data.columns.tolist()


class Topology(CustomDataFrame):
    """Topology dataframe

    Subclass of CustomDataFrame where the index is the receptors
    """
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
