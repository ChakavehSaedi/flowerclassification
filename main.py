"""
This code loads the MNIST database and the corresponding train and test subsets.
Trains a neural network
Shows the resutls
"""

import os

from data_prepare import data_handler
from model import net_handler
import argparse

################################################
# set up arguments
################################################
parser = argparse.ArgumentParser()
parser.add_argument("-inp", "--input_path", default="/data/input/")
parser.add_argument("-op", "--output_path", default="/data/output/")
parser.add_argument("-rld", "--reload_data", default=False)
parser.add_argument("-prt", "--pre_trained", default=False)
parser.add_argument("-e", "--epochs", default=15)
parser.add_argument("-bs", "--batch_size", default=4)
parser.add_argument("-cnn", "--cnn_info", default=[(3, 30, 5), (30, 60, 5)])
parser.add_argument("-ln", "--linear", default=[(10140, 120), (120, 84), (84, 5)])
parser.add_argument("-v", "--verbos", default=False)


args = parser.parse_args()
net_param = {}

net_param['cnn']=args.cnn_info
net_param['linear']=args.linear
epochs = args.epochs
batch_sise = args.batch_size

verbos = args.verbos
input_path = os.getcwd() + args.input_path
output_path = os.getcwd() + args.output_path
log_file = output_path + "log.txt"
reload_data = args.reload_data
pretrained = args.pre_trained

################################################
# Praparing data
################################################
print(" * Loading train and test data")
with open(log_file,'w') as log:
    log.write(" * Loading train and test data\n")

data_h = data_handler(reload_data, input_path, output_path, batch_sise)

################################################
# building the net
################################################
print(" * Building the netwrok")
with open(log_file,'a') as log:
    log.write(" * Building the netwrok\n")
network_h = net_handler(net_param, output_path, pretrained)

################################################
# train the net
################################################
if not pretrained:
    print(" * Training the netwrok")
    with open(log_file, 'a') as log:
        log.write(" * Training the netwrok\n")
    train_acc, train_loss = network_h.train_net(epochs, data_h.train_samples, data_h.val_samples, verbos, log_file)

################################################
# test the net
################################################
print(" * Testing the netwrok")
with open(log_file, 'a') as log:
    log.write(" * Testing the netwrok\n")
test_acc, train_loss = network_h.test_net(data_h.test_samples, log_file)


