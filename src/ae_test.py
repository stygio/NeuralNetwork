#!/usr/bin/python3

import network
import xlsParser

xlsParser.read_xls_iris('../data/IrisDataTrain.xls')
[x, y, z] = xlsParser.read_xls_iris('../data/IrisDataTrain.xls')

weight_range_l = -5
weight_range_h = 5
sig_choice = 1
eta = 1
loops = 300

ae = network.create_autoencoder(x, weight_range_l, weight_range_h, sig_choice, eta, loops)
