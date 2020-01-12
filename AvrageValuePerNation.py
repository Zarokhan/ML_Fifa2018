# https://www.kaggle.com/thec03u5/fifa-18-demo-player-dataset
# ['Name', 'Age', 'Nationality', 'Club', 'Value', 'Wage', 'Preferred Positions', 'Overall', 'Potential', 'Special', 'Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control', 'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing', 'Free kick accuracy', 'GK diving', 'GK handling', 'GK kicking', 'GK positioning', 'GK reflexes', 'Heading accuracy', 'Interceptions', 'Jumping', 'Long passing', 'Long shots', 'Marking', 'Penalties', 'Positioning', 'Reactions', 'Short passing', 'Shot power', 'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle', 'Strength', 'Vision', 'Volleys']

import pandas as pd
import numpy as np
from datetime import datetime

path = r"./data/CompleteDataset.csv"
headernames = ['Index', 'Name', 'Age', 'Photo', 'Nationality', 'Flag', 'Overall', 'Potential', 'Club', 'Club Logo', 'Value', 'Wage', 'Special', 'Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control', 'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing', 'Free kick accuracy', 'GK diving', 'GK handling', 'GK kicking', 'GK positioning', 'GK reflexes', 'Heading accuracy', 'Interceptions', 'Jumping', 'Long passing', 'Long shots', 'Marking', 'Penalties', 'Positioning', 'Reactions', 'Short passing', 'Shot power', 'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle', 'Strength', 'Vision', 'Volleys', 'CAM', 'CB', 'CDM', 'CF', 'CM', 'ID', 'LAM', 'LB', 'LCB', 'LCM', 'LDM', 'LF', 'LM', 'LS', 'LW', 'LWB', 'Preferred Positions', 'RAM', 'RB', 'RCB', 'RCM', 'RDM', 'RF', 'RM', 'RS', 'RW', 'RWB', 'ST']
data = pd.read_csv(path, names=headernames, index_col=0)
print(data.shape)

# Selection of data columns
sd = data[['Name', 'Age', 'Nationality', 'Club', 'Value', 'Wage', 'Preferred Positions']].copy()

# Clean dataset Value and Wage
sd['Unit'] = sd['Value'].str[-1]
sd['Value'] = np.where(sd['Unit'] == '0', 0, sd['Value'].str[1:-1].replace(r'[a-zA-Z]', ''))
sd['Value'] = sd['Value'].astype(float)
sd['Value'] = np.where(sd['Unit'] == 'M',  sd['Value']*1000000, sd['Value']*1000)
sd['Unit'] = sd['Wage'].str[-1]
sd['Wage'] = np.where(sd['Unit'] == '0', 0, sd['Wage'].str[1:-1].replace(r'[a-zA-Z]', ''))
sd['Wage'] = sd['Wage'].astype(float) * 1000
sd = sd.assign(Position=sd['Preferred Positions'].str.split().str[0])

#gb = sd.groupby(['Club']).Value.mean().reset_index().sort_values(by=['Value'], ascending = False)

gb = sd.groupby(['Nationality']).Value.mean().reset_index().sort_values(by=['Value'], ascending = False)
gb['Value'] = gb['Value'].astype(int)

print(gb.head())
print(gb.tail())
gb.to_excel("AvrageValuePerNation-" + datetime.now().strftime("%d-%m-%Y %H-%M-%S") + ".xlsx")

#print(data['Name'].count())
#result = footballers.groupby(['Nationality'], sort=False).CleanValue.mean().reset_index()
#result = result.sort_values(by=['CleanValue'])
