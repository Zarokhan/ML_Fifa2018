# Packages from here https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
from timeit import default_timer as timer
time_start = timer()

from matplotlib import pyplot
from datetime import datetime
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import StratifiedKFold
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn import preprocessing
import pandas
import numpy

names = ['ID', 'Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Position', 'Special', 'Wage', 'Value']
dataset = pandas.read_excel('./data/PlayerPersonalDataCleaned.xlsx', names=names, index_col=0, header=0).reset_index(drop=True)

feature_selection = ['Age', 'Nationality', 'Overall', 'Potential', 'Club', 'Position', 'Special', 'Wage', 'Value']
preprocess_classes = ['Nationality', 'Club', 'Position']
dataset = dataset[feature_selection].copy().dropna()

# pre process classes
for label in preprocess_classes:
    le = preprocessing.LabelEncoder()
    le.fit(dataset[label])
    dataset[label] = le.transform(dataset[label])
    # print(label)
    # print(list(le.classes_))
    # print(dataset[label])

# Split-out validation dataset
feature_size = len(feature_selection) - 1
features = dataset.values[:,0:feature_size]
predictor = dataset.values[:,feature_size]

# X = dataset.drop('Value', axis=1)
# y = dataset['Value']

# Testbed
testsize = 0.2
features_train, features_validation, predictor_train, predictor_validation = train_test_split(features, predictor, test_size=testsize, shuffle=True, random_state=1)

model = DecisionTreeClassifier()
regressor = model.fit(features_train, predictor_train)
predict_results = model.predict(features_validation)

meansquarederror = mean_squared_error(predictor_validation, predict_results)
meanabsoluteerror = mean_absolute_error(predictor_validation, predict_results)
accuracyscore = accuracy_score(predictor_validation, predict_results)
balancedeaccuracyscore = balanced_accuracy_score(predictor_validation, predict_results)
explainedvariancescore = explained_variance_score(predictor_validation, predict_results)
r2score = r2_score(predictor_validation, predict_results)

results = []
for i in range(len(predict_results)):
    results.append(predict_results[i] / predictor_validation[i])

threshold = 0.15
print(sum(1 for s in results if s > (1 - threshold) and s < (1 + threshold)) / len(results))

# end
time_end = timer()
print('\nTime: %s s \nAccuracy Score: %s\n' % ((time_end - time_start), accuracyscore))
print('balancedeaccuracyscore: %s' % (balancedeaccuracyscore))
print('meansquarederror: %s' % (meansquarederror))
print('meanabsoluteerror: %s' % (meanabsoluteerror))
print('explainedvariancescore: %s' % (explainedvariancescore))
print('r2score: %s\n' % (r2score))