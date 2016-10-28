require 'torch'
require 'nn'
-- global table deep structure neural network
dsnn = {}
dsnn.version = 1

unpack = unpack or table.unpack -- lua 5.2 compat

require('dsnn.Model')
