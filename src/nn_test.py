#!/usr/bin/python3

import network

path_to_data = '../data/'
training_file = 'Abalone.xlsx'
test_file = 'Abalone.xlsx'

file_training = path_to_data + training_file
file_test = path_to_data + test_file
weight_range_l = -0.1
weight_range_h = 0.1
sig_choice = 1
eta = 1
loops = 5
nn = network.initialize_network(file_training, weight_range_l, weight_range_h, sig_choice, eta, loops)
network.run_network(file_test, nn, 1)