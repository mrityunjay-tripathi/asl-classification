import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.preprocessing import LabelEncoder
#from tensorflow.python.framework import ops

import load_data, utils

np.random.seed(1)

### Load Data Set
(X_train, X_test), (Y_train, Y_test) = load_data.asl("data/")


### Encode given labels
la = LabelEncoder()
la.fit(Y_train)
Y_train_la = la.transform(Y_train)
Y_test_la = la.transform(Y_test)

### one-hot encoding of labels
n_classes = 29
Y_train = np.eye(n_classes)[Y_train_la]
Y_test = np.eye(n_classes)[Y_test_la]

### define parameters
m = X_train.shape[0]
lr = 0.01
num_epochs = 10
batch_size = 64
N = m//batch_size


### create tensorflow graph
with tf.name_scope('placeholders'):
    nH, nW, nC = 200, 200, 3
    nY = 29
    X = tf.placeholder(dtype=tf.float32, shape=[None, nH, nW, nC])
    Y = tf.placeholder(dtype=tf.float32, shape=[None, nY])

with tf.name_scope('forward_propagation'):

    W1 = tf.get_variable('W1', [4,4,3,8], initializer = tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable('W2', [2,2,8,16], initializer = tf.contrib.layers.xavier_initializer(seed=0))

    Z1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')

    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')

    P2 = tf.contrib.layers.flatten(P2) 
    Z3 = tf.contrib.layers.fully_connected(P2, num_outputs = 29, activation_fn = None)

with tf.name_scope('prediction'):
    y_hat = tf.argmax(Z3, 1)

with tf.name_scope('accuracy'):
    correct_pred = tf.equal(y_hat, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, "float"))

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))

with tf.name_scope('optimize'):
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)

with tf.name_scope('summaries'):
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged = tf.summary.merge_all()

### You will find the tensorflow graph at location '/tmp/asl-train'
train_writer = tf.summary.FileWriter('asl-train', tf.get_default_graph())


#tf.reset_default_graph()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    losses = []
    accuracies = []
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        for batch in utils.iterate_minibatches(X_train, Y_train):
            batch_x, batch_y = batch
            feed_dict = {X:batch_x, Y:batch_y}
            _, summary, LOSS, ACC = sess.run([train_op, merged, loss, accuracy], feed_dict = feed_dict)
            epoch_loss += LOSS/N
            epoch_accuracy += ACC/N
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)
        print(f"Epoch : {epoch+1}")
        print(f"Accuracy : {round(epoch_accuracy,4)}\tLoss : {round(epoch_loss,4)}")
        print()
        train_writer.add_summary(summary, epoch)
        
    ### save weights
    #utils.save_weights("asl-train", sess)

    plt.title(f"Learning Rate = {lr}")
    plt.xlabel("Epochs")
    plt.ylabel("Cost & Accuracy")
    plt.plot([i for i in range(num_epochs)], losses, accuracies)
    plt.show()
    
