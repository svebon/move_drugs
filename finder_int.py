import pandas as pd
from data_parsing import NamesMap
import yaml

'''
output_file = "C:/Users/Sveva/Desktop/interactive.csv"
output_folder = 'C:/Users/Sveva/Desktop/ML_datasets'
'''

### importo il file per creare il df1
input_one = 'C:/Users/mirco/Projects/Sveva/ML_datasets/drug.target.interaction.tsv'
df1 = pd.read_table(input_one)
df1 = df1[['DRUG_NAME', 'ACCESSION', 'GENE']]  # Così selezioni le colonne
# df1.columns = ['DRUG_NAME', 'ACCESSION']  # Così rinomini le colonne
df1['DRUG_NAME'] = df1['DRUG_NAME'].apply(lambda x: x.lower())
# df1 = df1.set_index('DRUG_NAME')
# print(df1)


### importo il file per creare il df2
input_two = 'C:/Users/mirco/Projects/Sveva/ML_datasets/new_mapped_name.csv'
df2 = pd.read_csv(input_two, sep=';')

names_D = list(df2['name_D'])
names_T = list(df2['name_T'])

common_names_D = [name_D for name_D in names_D if name_D in list(df1['ACCESSION'])]
common_names_T = [name_T for name_T in names_T if name_T in list(df1['GENE'])]

drugs_D = df1[df1['ACCESSION'].isin(common_names_D)][['DRUG_NAME', 'ACCESSION']]
drugs_T = df1[df1['GENE'].isin(common_names_T)][['DRUG_NAME', 'GENE']]

with open('paths.yaml') as f:
    paths = yaml.load(f, Loader=yaml.SafeLoader)

names_map = NamesMap()
names_map.load(paths['nodes'])

drugs_D['GENE'] = drugs_D['ACCESSION'].apply(lambda x: names_map.map_name(x))

drugs = pd.concat([drugs_D, drugs_T])
drugs = drugs[['DRUG_NAME', 'GENE']]

receptors = drugs['GENE'].unique()


drugs = drugs.set_index('GENE')

index = drugs['DRUG_NAME'].unique()

print(drugs)

interactions_lists = {rec: list(drugs.loc[rec]['DRUG_NAME']) for rec in receptors}

data = {rec: [int(drug in interactions) for drug in index] for rec, interactions in interactions_lists.items()}

print(data)


out = pd.DataFrame.from_dict(data, orient='columns')
out.index = index

print(out.to_string())

out.to_csv('P.csv')
