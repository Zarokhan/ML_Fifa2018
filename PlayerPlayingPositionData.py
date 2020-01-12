from matplotlib import pyplot
from datetime import datetime
import pandas


dataset = pandas.read_csv('./data/PlayerPlayingPositionData.csv', index_col=0, header=0).reset_index(drop=True)
print(dataset.shape)
print(dataset.head(dataset.shape[0]))

dataset.to_excel("PlayerPlayingPositionData-" + datetime.now().strftime("%d-%m-%Y %H-%M-%S") + ".xlsx")