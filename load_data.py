import numpy as np

def asl(path = ''):
    X_train = np.load(path + 'X_train.npy')
    X_test = np.load(path + 'X_test.npy')
    Y_train_orig = np.load(path + 'Y_train.npy')
    Y_train_orig = Y_train_orig.reshape((Y_train_orig.shape[0],))
    Y_test_orig = np.load(path + 'Y_test.npy')
    Y_test_orig = Y_test_orig.reshape((Y_test_orig.shape[0],))
    return (X_train, X_test), (Y_train_orig, Y_test_orig)