from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import numpy as np


def read_data(file_path):
    x_data, y_data =[],[]
    with open(file_path) as lines:
        for line in lines:
            x = line[:-1].split(' ')
            xx = [float(k) for k in x[:4]]
            yy = float(x[-1])
            x_data.append(xx)
            y_data.append(yy)
    x_train = np.array(x_data[:1500])
    y_train = np.array(y_data[:1500])
    x_test = np.array(x_data[1500:])
    y_test = np.array(y_data[1500:])
    return (x_train, y_train), (x_test, y_test)

def build_model():
    model = Sequential()
    model.add(Dense(units=10, input_dim=4))
    model.add(Activation("relu"))
    model.add(Dense(units=5))
    model.add(Activation("relu"))
    model.add(Dense(units=1))
    model.compile(loss="mean_absolute_error", optimizer=SGD(lr=0.0005, momentum=0.9))
    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = read_data("logistic_regression2.txt")
    model = build_model()
    model.fit(x_train, y_train, epochs=15, batch_size=64)
    loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
    classes = model.predict(x_test, batch_size=128)
    print(loss_and_metrics)
    #print(classes.shape)
    for i in range(20):
        print(classes[i], y_test[i])




