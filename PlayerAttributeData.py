from matplotlib import pyplot
from pandas import read_csv
from datetime import datetime

names = ['Index', 'Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control', 'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing', 'Free kick accuracy', 'GK diving', 'GK handling', 'GK kicking', 'GK positioning', 'GK reflexes', 'Heading accuracy', 'ID', 'Interceptions', 'Jumping', 'Long passing', 'Long shots', 'Marking', 'Penalties', 'Positioning', 'Reactions', 'Short passing', 'Shot power', 'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle', 'Strength', 'Vision', 'Volleys']
path = './data/PlayerAttributeData.csv'
dataset = read_csv(path, names=names, index_col=0)
print(dataset.shape)
print(dataset.head(dataset.shape[0]))

dataset.to_excel("PlayerAttributeData-" + datetime.now().strftime("%d-%m-%Y %H-%M-%S") + ".xlsx")