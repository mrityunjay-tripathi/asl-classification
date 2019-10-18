import tensorflow as tf
import numpy as np
import pickle

def iterate_minibatches(X, Y, batch_size = 64, shuffle = True):
    assert X.shape[0] == Y.shape[0]
    if shuffle:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
    for i in range(0, X.shape[0]-batch_size+1, batch_size):
        l = min(i + batch_size, X.shape[0])
        if shuffle:
            batch = indices[i:l]
        else:
            batch = slice(i,l)
        yield X[batch], Y[batch]


def save_weights(name, sess):
    tvars = tf.trainable_variables()
    tvars_vals = sess.run(tvars)
    parameters = {}
    for var, val in zip(tvars, tvars_vals):
        parameters[var.name] = val

    with open(name + '.txt', 'w') as f:
        f.write(pickle.dumps(parameters))