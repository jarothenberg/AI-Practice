import pickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    # p = u.load()
    # training_data, validation_data, test_data = pickle.load(f)
    training_data, validation_data, test_data = u.load()
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (1, 784)).tolist() for x in tr_d[0]]
    training_inputs = [element[0] for element in training_inputs]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    training_data = [[x[0], x[1]] for x in training_data]
    #print(training_data[0])

    validation_inputs = [np.reshape(x, (1, 784)).tolist() for x in va_d[0]]
    validation_inputs = [element[0] for element in validation_inputs]
    validation_data = list(zip(validation_inputs, va_d[1]))
    validation_data = [[x[0], [x[1]]] for x in validation_data]
    test_inputs = [np.reshape(x, (1, 784)).tolist() for x in te_d[0]]
    test_inputs = [element[0] for element in test_inputs]
    test_data = list(zip(test_inputs, te_d[1]))
    test_data = [[x[0], [x[1]]] for x in test_data]
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    e[j] = 1.0
    return e
