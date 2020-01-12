# Packages from here https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
from matplotlib import pyplot
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas
import numpy
import math

names = ['ID', 'Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control', 'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing', 'Free kick accuracy', 'GK diving', 'GK handling', 'GK kicking', 'GK positioning', 'GK reflexes', 'Heading accuracy', 'Interceptions', 'Jumping', 'Long passing', 'Long shots', 'Marking', 'Penalties', 'Positioning', 'Reactions', 'Short passing', 'Shot power', 'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle', 'Strength', 'Vision', 'Volleys', 'Value']
dataset = pandas.read_excel('./data/FIFA_ML_PlayerAttributeData.xlsx', names=names, index_col=0, header=0).reset_index(drop=True)

feature_selection = ['Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control', 'Composure', 'Crossing', 'Curve', 'Dribbling', 'Finishing', 'Free kick accuracy', 'GK diving', 'GK handling', 'GK kicking', 'GK positioning', 'GK reflexes', 'Heading accuracy', 'Interceptions', 'Jumping', 'Long passing', 'Long shots', 'Marking', 'Penalties', 'Positioning', 'Reactions', 'Short passing', 'Shot power', 'Sliding tackle', 'Sprint speed', 'Stamina', 'Standing tackle', 'Strength', 'Vision', 'Volleys', 'Value']
preprocess_classes = []
dataset = dataset[feature_selection].copy().dropna()
print(dataset.shape)

# pre process classes
for label in preprocess_classes:
    le = preprocessing.LabelEncoder()
    le.fit(dataset[label])
    dataset[label] = le.transform(dataset[label])

# scatter matrix
# pandas.plotting.scatter_matrix(dataset, alpha=0.5)
# pyplot.show()
# quit()

# Split-out validation dataset
array = dataset.values

feature_size = len(feature_selection) - 1
features = array[:,0:feature_size]
predictor = array[:,feature_size]

testsize = 0.2
features_train, features_validation, predictor_train, predictor_validation = train_test_split(features, predictor, test_size=testsize, shuffle=True, random_state=1)

print(features_train.shape)
print(features_train)

print(features_validation.shape)
print(features_validation)

print(predictor_train.shape)
print(predictor_train)

print(predictor_validation.shape)
print(predictor_validation)

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
validationResults = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
	cross_val_results = cross_val_score(model, features_train, predictor_train, cv=kfold, scoring='accuracy')
	model.fit(features_train, predictor_train)
	predict_results = model.predict(features_validation)
	results.append(cross_val_results)
	validationResults.append(predict_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cross_val_results.mean(), cross_val_results.std()))
	accuracyscore = accuracy_score(predictor_validation, predict_results)
	print('Accuracy Score: %s' % (accuracyscore))
# Compare Algorithms
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()






# pre process dataset (handle + and -)
# for label in feature_selection:
# 	plussplit = dataset[label].str.split('+', 1, True)
# 	minussplit = dataset[label] = dataset[label].str.split('-', 1, True)
# 	for x in range(len(plussplit)):
# 		if (plussplit.values[x][1]):
# 			print(plussplit.values[x].astype(float))
# 			dataset[label].values[x] = plussplit.values[x].astype(float).sum()
# 			print(dataset[label].values[x])
# 	for x in range(len(minussplit)):
# 		if (minussplit.values[x][1]):
# 			# print(minussplit.values[x].astype(float))
# 			dataset[label].values[x] = float(dataset[label].values[x]) - minussplit.values[x].astype(float)[1]
# 			# print(dataset[label].values[x])
# dataset = dataset.astype(float)
# dataset.to_excel("FIFA_ML_PlayerAttributeData-" + datetime.now().strftime("%d-%m-%Y %H-%M-%S") + ".xlsx")
