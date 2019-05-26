#!/usr/bin/python3

import network

path_to_data = '../data/'
training_file = 'IrisDataTrain.xls'
test_file = 'IrisDataTest.xls'

file_training = path_to_data + training_file
file_test = path_to_data + test_file
weight_range_l = -1
weight_range_h = 1
sig_choice = 1
eta = 0.5
ae_loops = 250
tr_loops = 100
nn = network.initialize_autoencoder_mlp(file_training, weight_range_l, weight_range_h, sig_choice, eta, ae_loops, tr_loops)
network.run_mlp(file_test, nn, sig_choice)