from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.normalization import batch_normalization
import numpy as np
import tensorflow as tf
import math
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

step_radians = 0.01
steps_of_history = 20
steps_in_future = 1
index = 0



def get_data(x):
    seq = []
    next_val = []

    for i in range(0, len(x) - steps_of_history, steps_in_future):
        seq.append(x[i: i + steps_of_history])
        next_val.append(x[i + steps_of_history])

    seq = np.reshape(seq, [-1, steps_of_history, 1])
    next_val = np.reshape(next_val, [-1, 1])
    print(np.shape(seq))

    trainX = np.array(seq)
    trainY = np.array(next_val)
    return trainX, trainY


def myRNN(x, activator, optimizer, domain=""):
    trainX, trainY = get_data(x)
    tf.reset_default_graph()
    # Network building
    net = tflearn.input_data(shape=[None, steps_of_history, 1])
    net = tflearn.lstm(net, 32, dropout=0.8, bias=True)
    net = tflearn.fully_connected(net, 1, activation=activator)

    net = tflearn.regression(net, optimizer=optimizer, loss='mean_square')
    # customize==>
    #sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=100)
    #network = tflearn.regression(net, optimizer=sgd, loss='mean_square')
    
    model = tflearn.DNN(net)

    """
    net = tflearn.input_data(shape=[None, steps_of_history, 1])
    net = tflearn.simple_rnn(net, n_units=32, return_seq=False)
    net = tflearn.fully_connected(net, 1, activation='linear')
    net = tflearn.regression(net, optimizer='sgd', loss='mean_square', learning_rate=0.1)

    # Training
    model = tflearn.DNN(net, clip_gradients=0.0, tensorboard_verbose=0)
    """

    model.fit(trainX, trainY, n_epoch=100, validation_set=0.1, batch_size=128)

    # Testing
    #x = np.sin(np.arange(20*math.pi, 24*math.pi, step_radians))

    seq = []
    Y = []

    for i in range(0, len(x) - steps_of_history, steps_in_future):
        seq.append(x[i: i + steps_of_history])
        Y.append(x[i + steps_of_history])

    seq = np.reshape(seq, [-1, steps_of_history, 1])
    testX = np.array(seq)

    # Predict the future values
    predictY = model.predict(testX)
    print(predictY)

    # Plot the results
    plt.figure(figsize=(20,4))
    plt.suptitle('Prediction')
    plt.title('History='+str(steps_of_history)+', Future='+str(steps_in_future))
    plt.plot(Y, 'r-', label='Actual')
    plt.plot(predictY, 'gx', label='Predicted')
    plt.legend()
    if domain:
        plt.savefig('pku_'+activator+"_"+optimizer+"_"+domain+".png")
    else:
        plt.savefig('pku_'+activator+"_"+optimizer+".png")


def main():
    out_data = {}
    with open("out_data.json") as f:
        out_data = json.load(f)
        #x = out_data["www.coe.pku.edu.cn"]
        #myRNN(x, activator="prelu", optimizer="rmsprop", domain="www.coe.pku.edu.cn")
    # I find that prelu and rmsprop is best choice
    x = out_data["www.coe.pku.edu.cn"]
    activators = ['linear', 'tanh', 'sigmoid', 'softmax', 'softplus', 'softsign', 'relu', 'relu6', 'leaky_relu', 'prelu', 'elu']
    optimizers = ['sgd', 'rmsprop', 'adam', 'momentum', 'adagrad', 'ftrl', 'adadelta']
    for activator in activators:   
        for optimizer in optimizers:
            print ("Running for : "+ activator + " & " + optimizer)
            myRNN(x, activator, optimizer)
main()