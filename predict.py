import os, sys, argparse
import numpy as np
import tensorflow as tf
from PIL import Image


def predict(image_location):
    tf.reset_default_graph()
    tf.set_random_seed(1)
    img = Image.open(image_location)
    img = img.resize((200,200))
    x_test = np.asarray(img)/255.0
    x_test = x_test.astype(np.float32)
    x_test = x_test.reshape(1, *x_test.shape)


    with tf.Session() as sess:    
        saver = tf.train.import_meta_graph('saved_model/asl-model-10.meta')
        saver.restore(sess,tf.train.latest_checkpoint('saved_model/'))
        graph = tf.get_default_graph()
        W1 = graph.get_tensor_by_name('W1:0')
        W2 = graph.get_tensor_by_name('W2:0')

    nH, nW, nC = 200, 200, 3
    X = tf.placeholder(name = 'X', dtype=tf.float32, shape=[None, nH, nW, nC])

    Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')

    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')

    P2 = tf.contrib.layers.flatten(P2) 
    Z3 = tf.contrib.layers.fully_connected(P2, num_outputs = 29, activation_fn = None)

    y_hat = tf.argmax(Z3, 1)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)
        
        feed_dict = {X:x_test}
        index = sess.run(y_hat, feed_dict = feed_dict)[0]
        return index


if __name__ == '__main__':
    img_label = predict("images/space_test.jpg")
    print(img_label)