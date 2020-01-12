from matplotlib import pyplot
from pandas import read_csv
from datetime import datetime

path = r"./data/CompleteDataset.csv"
headernames = ['Index', 'Name', 'Age', 'Photo', 'Nationality', 'Flag', 'Overall', 'Potential', 'Club', 'Club Logo', 'Value', 'Wage', 'Special', 'Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control', 'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing', 'Free kick accuracy', 'GK diving', 'GK handling', 'GK kicking', 'GK positioning', 'GK reflexes', 'Heading accuracy', 'Interceptions', 'Jumping', 'Long passing', 'Long shots', 'Marking', 'Penalties', 'Positioning', 'Reactions', 'Short passing', 'Shot power', 'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle', 'Strength', 'Vision', 'Volleys', 'CAM', 'CB', 'CDM', 'CF', 'CM', 'ID', 'LAM', 'LB', 'LCB', 'LCM', 'LDM', 'LF', 'LM', 'LS', 'LW', 'LWB', 'Preferred Positions', 'RAM', 'RB', 'RCB', 'RCM', 'RDM', 'RF', 'RM', 'RS', 'RW', 'RWB', 'ST']
data = read_csv(path, names=headernames, index_col=0)
print(data.shape)
#print(data.head(50))
#print(data.head(data.shape[0]))

# Histograms group the data in bins and is the fastest way to get idea about the distribution of each attribute in dataset.
def listHistograms():
    data.hist()
    pyplot.show()
    pass

import re
import numpy as np

footballers = data.copy()
footballers['Unit'] = data['Value'].str[-1]
footballers['Value (M)'] = np.where(footballers['Unit'] == '0', 0, footballers['Value'].str[1:-1].replace(r'[a-zA-Z]', ''))
footballers['Value (M)'] = footballers['Value (M)'].astype(float)
footballers['Value (M)'] = np.where(footballers['Unit'] == 'M',  footballers['Value (M)'], footballers['Value (M)']/1000)
footballers = footballers.assign(Value=footballers['Value (M)'], Position=footballers['Preferred Positions'].str.split().str[0])

print(footballers.head())

import seaborn as sns

df = footballers[footballers['Position'].isin(['ST', 'GK'])]
g = sns.FacetGrid(df, col="Position")
g.map(sns.kdeplot, "Overall")

footballers.to_excel("CompleteDataset-" + datetime.now().strftime("%d-%m-%Y %H-%M-%S") + ".xlsx")