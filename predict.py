import tensorflow as tf
import pandas as pd
import time
import os

# initializing variables
dropout = 1
n_classes = 10

# defining variables
x = tf.placeholder('float',[None,28,28,1], name='x')

# returns the training data after formatting
def get_data(index=1):
    df_test = pd.read_csv('test.csv')

    X_test = df_test.values.reshape(-1,784)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_test = X_test.astype('float32')
    X_test /= 255

    if index==1:
        return X_test[0:7000]
    if index==2:
        return X_test[7000:14000]
    if index==3:
        return X_test[14000:21000]
    if index==4:
        return X_test[21000:28000]

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def cnn_model(x):
	# initializing weights
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([7*7*64,128])),
               'out':tf.Variable(tf.random_normal([128, n_classes]))}
    
    # initializing biases
    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
              'b_conv2':tf.Variable(tf.random_normal([64])),
              'b_fc':tf.Variable(tf.random_normal([128])),
              'out':tf.Variable(tf.random_normal([n_classes]))}
        
    # reshaping x to image format
    x = tf.reshape(x, shape=[-1,28,28,1])
        
    # first convolution layer 
    conv1 = conv2d(x, weights['W_conv1']) + biases['b_conv1']
    conv1 = tf.nn.relu(conv1)
    conv1 = maxpool2d(conv1)

    # second convolution layer 
    conv2 = conv2d(conv1, weights['W_conv2']) + biases['b_conv2']
    conv2 = tf.nn.relu(conv2)
    conv2 = maxpool2d(conv2)
    
    # fully connected layer
    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.matmul(fc, weights['W_fc']) + biases['b_fc']
    fc = tf.nn.relu(fc)

    # dropout layer
    fc = tf.nn.dropout(fc, dropout)
    
    # output layer
    output = tf.matmul(fc, weights['out']) + biases['out']

    return output

def train_model(x):
	# getting output after feed forward
    prediction = cnn_model(x)

    # defining session
    with tf.Session() as sess:
    	# initializing variables
        sess.run(tf.global_variables_initializer())

        # creating a saver object to restore previous weights
        saver = tf.train.Saver()

        pwd = os.getcwd()
        os.chdir(pwd+'/tmp')
        # loading latest weights
        saver.restore(sess,tf.train.latest_checkpoint('./'))
        os.chdir(pwd)

        out = tf.argmax(prediction,1)

        # getting data
        X_test = get_data(1)
        pred = out.eval({x: X_test})
        # creating Series to store predictions
        ser = pd.Series(pred, index=range(1,7001))
        print('1st set completed')

        X_test = get_data(2)
        pred = out.eval({x: X_test})
        ser = ser.append(pd.Series(pred, index=range(7001,14001)))
        print('2nd set completed')

        X_test = get_data(3)
        pred = out.eval({x: X_test})
        ser = ser.append(pd.Series(pred, index=range(14001,21001)))
        print('3rd set completed')

        X_test = get_data(4)
        pred = out.eval({x: X_test})
        ser = ser.append(pd.Series(pred, index=range(21001,28001)))
        print('4th set completed')

        # saving predictions
        ser.to_csv('output.csv')
        print('File saved')
train_model(x)