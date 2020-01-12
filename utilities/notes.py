from pandas import read_csv

path = r"./data/CompleteDataset.csv"
headernames = ['Index', 'Name', 'Age', 'Photo', 'Nationality', 'Flag', 'Overall', 'Potential', 'Club', 'Club Logo', 'Value', 'Wage', 'Special', 'Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control', 'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing', 'Free kick accuracy', 'GK diving', 'GK handling', 'GK kicking', 'GK positioning', 'GK reflexes', 'Heading accuracy', 'Interceptions', 'Jumping', 'Long passing', 'Long shots', 'Marking', 'Penalties', 'Positioning', 'Reactions', 'Short passing', 'Shot power', 'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle', 'Strength', 'Vision', 'Volleys', 'CAM', 'CB', 'CDM', 'CF', 'CM', 'ID', 'LAM', 'LB', 'LCB', 'LCM', 'LDM', 'LF', 'LM', 'LS', 'LW', 'LWB', 'Preferred Positions', 'RAM', 'RB', 'RCB', 'RCM', 'RDM', 'RF', 'RM', 'RS', 'RW', 'RWB', 'ST']
data = read_csv(path, names=headernames, index_col=0)

data.info()

# pandas.plotting.scatter_matrix(dataset)
# pyplot.show()

# Value distrubution over these classes
# Age/Value

# ageValueDataset = dataset[['Age', 'Value']].copy()
# print(ageValueDataset.shape)
# print(ageValueDataset.head(ageValueDataset.shape[0]))
# pandas.plotting.scatter_matrix(ageValueDataset)
# ageValueDataset.plot(x='Age', y='Value', style='o')
# ageValueDataset.hist()
# pyplot.show()

# Nationality/Value
# Overall/Value
# Potential/Value
# Club/Value
# Wage/Value
# Special/Value
# Position/Value

# print('X train')
# print(X_train.values)
# print('X validation')
# print(X_validation.values)
# print('Y train')
# print(Y_train.values)
# print('Y validation')
# print(Y_validation.values)

# print('trainset shape')
# print(trainset.shape)
# print('testset shape')
# print(testset.shape)

# timenow = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
# trainset.to_excel("FIFA_ML-trainset-" + str((1.0 - testsize) * 100) + "-" + timenow + ".xlsx")
# # testset.to_excel("FIFA_ML-testset-" + str(testsize * 100) + "-" + timenow + ".xlsx")

# model = GaussianNB()
# model.fit()
# predictions = model.predict(testset)
# print(predictions)

# # Split-out validation dataset
# print(dataset.values)