import numpy as np

def getTrainTestVal(temp_train, temp_test, temp_val):

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    x_val = []
    y_val = []

    for x in temp_train:
        x_train.append(x[0])
        y_train.append(x[1])

    for x in temp_test:
        x_test.append(x[0])
        y_test.append(x[1])

    for x in temp_val:
        x_val.append(x[0])
        y_val.append(x[1])

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    x_val = np.array(x_val)
    y_val = np.array(y_val)

    return x_train, y_train, x_test, y_test, x_val, y_val