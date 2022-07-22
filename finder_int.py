import pandas as pd
from data_parsing import NamesMap

interactions_file = 'C:/Users/mirco/Projects/Sveva/ML_datasets/drug.target.interaction.tsv'
receptors_file = 'C:/Users/mirco/Projects/Sveva/ML_datasets/new_mapped_name.csv'

output_file = 'interactive.csv'

# Interactions' DataFrame parsing

interactions_df = pd.read_table(interactions_file)
interactions_df = interactions_df[['DRUG_NAME', 'ACCESSION', 'GENE']]
interactions_df['DRUG_NAME'] = interactions_df['DRUG_NAME'].apply(lambda x: x.lower())

# Receptors' DataFrame parsing
receptors_df = pd.read_csv(receptors_file, sep=';')
names_D = list(receptors_df['name_D'])
names_T = list(receptors_df['name_T'])

# Intersection between receptors' and interactions' DataFrames
common_names_D = [name_D for name_D in names_D if name_D in list(interactions_df['ACCESSION'])]
common_names_T = [name_T for name_T in names_T if name_T in list(interactions_df['GENE'])]

names_map = NamesMap()
names_map.load(receptors_file)

drugs_D = interactions_df[interactions_df['ACCESSION'].isin(common_names_D)][['DRUG_NAME', 'ACCESSION']]
drugs_D['GENE'] = drugs_D['ACCESSION'].apply(lambda x: names_map.map_name(x))

drugs_T = interactions_df[interactions_df['GENE'].isin(common_names_T)][['DRUG_NAME', 'GENE']]


# P creation

drugs = pd.concat([drugs_D, drugs_T])
drugs = drugs[['DRUG_NAME', 'GENE']]

P_index = drugs['DRUG_NAME'].unique()
P_cols = drugs['GENE'].unique()

drugs = drugs.set_index('GENE')

interactions_lists = {rec: list(drugs.loc[rec]['DRUG_NAME']) for rec in P_cols}

data = {rec: [int(drug in interactions) for drug in P_index] for rec, interactions in interactions_lists.items()}

print(data)

P = pd.DataFrame.from_dict(data, orient='columns')
P.index = P_index

print(P.to_string())

print(P.describe().to_string())

P.to_csv(output_file)
