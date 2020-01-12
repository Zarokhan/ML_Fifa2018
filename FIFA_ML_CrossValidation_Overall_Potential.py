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

names = ['ID', 'Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Position', 'Special', 'Wage', 'Value']
dataset = pandas.read_excel('./data/PlayerPersonalDataCleaned.xlsx', names=names, index_col=0, header=0).reset_index(drop=True)

feature_selection = ['Wage', 'Potential', 'Value']
preprocess_classes = []
dataset = dataset[feature_selection].copy().dropna()

# pre process classes
for label in preprocess_classes:
    le = preprocessing.LabelEncoder()
    le.fit(dataset[label])
    dataset[label] = le.transform(dataset[label])

# dataset.to_excel("FIFA_ML_CrossValidation_Overall_Potential-" + datetime.now().strftime("%d-%m-%Y %H-%M-%S") + ".xlsx")
# quit()

# scatter matrix
# pandas.plotting.scatter_matrix(dataset, alpha=0.5)
# pyplot.show()

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
models.append(('SVM', SVC(gamma='scale')))
# evaluate each model in turn
results = []
validationResults = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
	cross_val_results = cross_val_score(model, features_train, predictor_train, cv=5, scoring='accuracy')
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