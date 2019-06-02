from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pickle
import numpy as np


PATH_TO_DATA = '../data/data.csv'
RANDOM_STATE = 42
SAVE_STATS_PATH = './stats'
SAVE_MODEL_PATH = '../trainer'


def labels_to_y_bin(y):
    return [1 if target == 'M' else 0 for target in y]

# read data
df = pd.read_csv(PATH_TO_DATA, sep=',')

# preprocess data
y = df['diagnosis'].values
df = df.drop(['id', 'diagnosis'], axis=1)
df = df.fillna(value=df.mean())
# df_norm = (df - df.mean())/df.std(ddof=0)
scaler = StandardScaler()
X = scaler.fit_transform(df)

# naive Bayes
naive_bayes = GaussianNB()

# logistic regression
logistic_regression = LogisticRegression(solver='lbfgs', max_iter=2000, random_state=RANDOM_STATE)

# svm
support_vector_machine = svm.SVC(kernel='rbf', C=1.0, gamma='scale', random_state=RANDOM_STATE)

# random forest
random_forest = RandomForestClassifier(n_estimators=100, max_depth=4, criterion='entropy', random_state=RANDOM_STATE)

# xgboost
gradient_boosting = XGBClassifier(max_depth=4, n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE)

# MLP
mlp = MLPClassifier(hidden_layer_sizes=10, max_iter=2000, random_state=RANDOM_STATE)

y_bin = labels_to_y_bin(y)
all_models = [naive_bayes, logistic_regression, support_vector_machine, random_forest, gradient_boosting, mlp]
all_names = ['naive_bayes', 'logistic_regression', 'support_vector_machine', 'random_forest', 'gradient_boosting', 'mlp']
scoring = {'accuracy': 'accuracy', 'f1_score': 'f1'}

np.random.seed(RANDOM_STATE)
cv_results = []

for i in range(len(all_models)):
    cv_results.append(cross_validate(all_models[i], X, y_bin, cv=10, scoring=scoring, return_train_score=False))

test_accuracy_templates = []
for i in range(len(cv_results)):
    results = cv_results[i]['test_accuracy']*100
    test_accuracy_templates.append('Model number: {} || Model: {} || Min: {} || Max: {} || Mean: {} || Median: {} || Standard deviation: {}'.format(
        i + 1, all_names[i], min(results), max(results), np.mean(results), np.median(results), np.std(results)))

test_accuracy_template = '\n'.join(test_accuracy_templates)
print('Accuracy')
print(test_accuracy_template)
with open(SAVE_STATS_PATH + '/test_accuracy', 'w') as test_accuracy_file:
    test_accuracy_file.write(test_accuracy_template)

test_f1_score_templates = []
for i in range(len(cv_results)):
    results = cv_results[i]['test_f1_score']
    test_f1_score_templates.append('Model number: {} || Model: {} || Min: {} || Max: {} || Mean: {} || Median: {} || Standard deviation: {}'.format(
        i + 1, all_names[i], min(results), max(results), np.mean(results), np.median(results), np.std(results)
    ))

test_f1_score_template = '\n'.join(test_f1_score_templates)
print('F1_score')
print(test_f1_score_template)
with open(SAVE_STATS_PATH + '/test_f1_score', 'w') as test_f1_score_file:
    test_f1_score_file.write(test_f1_score_template)

print()
model_to_save = int(input('Select the model to save by entering its number '))

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=RANDOM_STATE)
y_train_bin = labels_to_y_bin(y_train)
y_test_bin = labels_to_y_bin(y_test)
final_model = all_models[model_to_save - 1].fit(X_train, y_train_bin)
final_model_test = final_model.predict(X_test)

final_accuracy = 100*accuracy_score(y_test_bin, final_model_test)
print('Final model accuracy on test set: {}'.format(final_accuracy))
with open(SAVE_STATS_PATH + '/final_test_accuracy', 'w') as final_test_accuracy_file:
    final_test_accuracy_file.write(str(final_accuracy))

final_f1_score = f1_score(y_test_bin, final_model_test)
print('Final model F1_score on test set: {}'.format(final_f1_score))
with open(SAVE_STATS_PATH + '/final_test_f1_score', 'w') as final_test_f1_score_file:
    final_test_f1_score_file.write(str(final_f1_score))

pickle.dump(final_model, open(SAVE_MODEL_PATH + '/final_model_{}.sav'.format(all_names[model_to_save - 1]), 'wb'))
