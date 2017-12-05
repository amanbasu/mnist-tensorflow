import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import os

# initializing variables
dropout = 0.8
n_classes = 10
batch_size = 128
hm_epochs = 1000
learning_rate = 0.0015

# defining variables
x = tf.placeholder('float',[None,28,28,1], name='x')
y = tf.placeholder('float', name='y')

# returns the training data after formatting
def get_data():
    df_train = pd.read_csv('train.csv')

    # separating labels and features
    X_train = df_train.drop('label', axis=1)
    y_train = df_train['label']

    # normalizing input features
    X_train = X_train.values.reshape(-1,784)
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_train = X_train.astype('float32')
    X_train /= 255

    # changing labels to one hot encoding
    y_train = tf.contrib.keras.utils.to_categorical(y_train, 10)

    # splitting input to train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, random_state=None, test_size=0.1)

    return [X_train, X_test, y_train, y_test]

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

    # calculating cross entropy loss
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    # defining optimizer 
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    # getting the input data
    [X_train, X_test, y_train, y_test] = get_data()

    # total number of batches
    hm_batches = int(X_train.shape[0]/batch_size)
    
    # defining session
    with tf.Session() as sess:
    	# initializing variables
        sess.run(tf.global_variables_initializer())

        # creating a saver object to restore previous weights
        saver = tf.train.Saver()

        pwd = os.getcwd()
        os.chdir(pwd+'/tmp')
        # saver = tf.train.import_meta_graph("mnist_model_1512391103.meta")
        # loading latest weights
        saver.restore(sess,tf.train.latest_checkpoint('./'))
        

        # training
        for epochs in range(hm_epochs):
            epoch_loss = 0
            t1 = time.time()
            for i in range(hm_batches):
                batch_x = X_train[i*batch_size:(i+1)*batch_size]
                batch_y = y_train[i*batch_size:(i+1)*batch_size]
                
                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
                epoch_loss += c / hm_batches
            
            t2 = time.time()
            print('Epoch:',epochs+1,'/',hm_epochs,'Loss:',epoch_loss,'Time: %.2fs'%(t2-t1))

        # saving weights
        saver.save(sess, 'mnist_model_'+ str(round(time.time())))
        os.chdir(pwd)

        # getting accuracy
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy: %.4f%%' % (accuracy.eval({x: X_test, y: y_test})*100))
train_model(x)
