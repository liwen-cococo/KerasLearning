import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_score
from sklearn.cross_validation import train_test_split

def main():
    X, y = convert_data('car.data.txt')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'max_depth':4, 'eta':0.3, 'objective':'multi:softprob', 'num_class':4}
    num_round = 20
    bst = xgb.train(param, dtrain, num_round)

    preds = bst.predict(dtest)
    best_preds = np.asarray([np.argmax(line) for line in preds])

    score = precision_score(y_test, best_preds, average='macro')
    print(score)


def convert_format(line):
    d_buying = {'vhigh':0, 'high':1, 'med':2, 'low':3}
    d_maint = {'vhigh':0, 'high':1, 'med':2, 'low':3}
    d_doors = {'2':0, '3':1, '4':2, '5more':3}
    d_persons = {'2':0, '4':1, 'more':2}
    d_lug_boot = {'small':0, 'med':1, "big":2}
    d_safety = {'low':0, 'med':1, "high":2}
    d_class = {'unacc':0, 'acc':1, 'good':2, 'vgood':3}
    d = line.split(',')
    
    x = [d_buying[d[0]], d_maint[d[1]], d_doors[d[2]],
        d_persons[d[3]], d_lug_boot[d[4]], d_safety[d[5]]]
    y = d_class[d[6][:-1]]

    return (x, y)


def convert_data(file_path):
    data_X, data_y = [], []

    with open(file_path) as lines:
        for line in lines:
            (X, y) = convert_format(line)
            data_X.append(X)
            data_y.append(y)
    
    return (np.array(data_X), np.array(data_y))


if __name__ == '__main__':
    main()

