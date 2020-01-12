from matplotlib import pyplot
from datetime import datetime
import pandas
import numpy

# names = ['ID', 'Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Value', 'Wage', 'Special']
dataset = pandas.read_csv('./data/PlayerPersonalData.csv', index_col=0, header=0).reset_index(drop=True)
print(dataset.shape)
# print(dataset.describe())
# print(dataset.head(dataset.shape[0]))

dataset['Unit'] = dataset['Value'].str[-1]
dataset['Value'] = numpy.where(dataset['Unit'] == '0', 0, dataset['Value'].str[1:-1].replace(r'[a-zA-Z]', ''))
dataset['Value'] = dataset['Value'].astype(float)
dataset['Value'] = numpy.where(dataset['Unit'] == 'M',  dataset['Value']*1000000, dataset['Value']*1000)
dataset['Unit'] = dataset['Wage'].str[-1]
dataset['Wage'] = numpy.where(dataset['Unit'] == '0', 0, dataset['Wage'].str[1:-1].replace(r'[a-zA-Z]', ''))
dataset['Wage'] = dataset['Wage'].astype(float) * 1000
#dataset = dataset.assign(Position=dataset['Preferred Positions'].str.split().str[0])

dataset.to_excel("PlayerPersonalData-" + datetime.now().strftime("%d-%m-%Y %H-%M-%S") + ".xlsx")