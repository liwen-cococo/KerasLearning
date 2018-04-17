from sklearn import datasets
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import numpy as np
from sklearn.metrics import precision_score

iris = datasets.load_iris()
X = iris.data
y = iris.target
print('type(X) = ', type(X))
print('type(y) = ', type(y))
print('X.shape = ', X.shape)
print('y.shape = ', y.shape)
for i in range(110, 130):
    print(X[i], y[i])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('type(X_train) = ', type(X_train))
print('type(y_train) = ', type(y_train))
print('X_train.shape = ', X_train.shape)
print('y_train.shape = ', y_train.shape)
print(X_train, '\n')
print(X_test, '\n')
print(y_train, '\n')
print(y_test, '\n')

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

param = {'max_depth':3, 'eta':0.3, 'objective':'multi:softprob', 'num_class':3}
num_round = 20


bst = xgb.train(param, dtrain, num_round)
bst.dump_model('dump.raw.txt')

preds = bst.predict(dtest)
best_preds = np.asarray([np.argmax(line) for line in preds])



