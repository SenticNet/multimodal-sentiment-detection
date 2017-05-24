"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""

import morb
from morb import rbms, stats, updaters, trainers, monitors, units, parameters

import theano
import theano.tensor as T
import csv
import numpy as np

import gzip, cPickle, time

import matplotlib.pyplot as plt
plt.ion()

from utilsm import generate_data, get_context

# DEBUGGING

from theano import ProfileMode
# mode = theano.ProfileMode(optimizer='fast_run', linker=theano.gof.OpWiseCLinker())
# mode = theano.compile.DebugMode(check_py_code=False, require_matching_strides=False)
mode = None

#############################################################################
from sklearn.decomposition import PCA


import os
import sys
import timeit
import cPickle
import pickle
import numpy
import numpy.core.multiarray
import csv

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

neuron = 10;
layer0gW = 0;
layer1gW = 0;
layer1bgW = 0;
layer1cgW = 0;
layer1dgW =0;
layer1egW = 0;
layer1fgW = 0;
layer1ggW = 0;
layer1hgW = 0;
layer2gW = 0;
layer3gW = 0;
layer0gb = 0;
layer1gb = 0;
layer1bgb = 0;
layer1cgb = 0;
layer1dgb = 0;
layer1egb = 0;
layer1fgb = 0;
layer1ggb = 0;
layer1hgb = 0;
layer2gb = 0;
layer3gb = 0;
lay2w = 0;
all_test = 0;
eval_print1 = 0;
eval_print2 = 0;
eval_print3 = 0;
epoch_cd = 0;
batchm = 0;

def whiten(data):

  pca = PCA(whiten=True)
  transformed = pca.fit_transform(data)
  pca.whiten = False
  zca = pca.inverse_transform(transformed)
  
  return zca

def morbrun1(f1=1, f2=1, v1=1, v2=1, kern = 1):
      
  test_set_x = np.array(eval_print1).flatten(2)
  valid_set_x = np.array(eval_print3).flatten(2)
  train_set_x = np.array(eval_print2).flatten(2)

  train_set_x = train_set_x.reshape(np.array(eval_print2).shape[0]*batchm,kern,v1,v2)
  valid_set_x = valid_set_x.reshape(np.array(eval_print3).shape[0]*batchm,kern,v1,v2)   
  test_set_x = test_set_x.reshape(np.array(eval_print1).shape[0]*batchm,kern,v1,v2)

  visible_maps = kern
  hidden_maps = neuron # 100 # 50
  filter_height = f1 # 7 # 8
  filter_width = f2 # 30 # 8
  mb_size = batchm # 1 minibatch
  
  print ">> Constructing RBM..."
  fan_in = visible_maps * filter_height * filter_width

  """
   initial_W = numpy.asarray(
            self.numpy_rng.uniform(
                low = - numpy.sqrt(3./fan_in),
                high = numpy.sqrt(3./fan_in),
                size = self.filter_shape
            ), dtype=theano.config.floatX)
  """
  numpy_rng = np.random.RandomState(123)
  initial_W = np.asarray(
            numpy_rng.normal(
                0, 0.5 / np.sqrt(fan_in),
                size = (hidden_maps, visible_maps, filter_height, filter_width)
            ), dtype=theano.config.floatX)
  initial_bv = np.zeros(visible_maps, dtype = theano.config.floatX)
  initial_bh = np.zeros(hidden_maps, dtype = theano.config.floatX)



  shape_info = {
   'hidden_maps': hidden_maps,
   'visible_maps': visible_maps,
   'filter_height': filter_height,
   'filter_width': filter_width,
   'visible_height': v1, #45+8,
   'visible_width': v2, #30,
   'mb_size': mb_size
  }

  # rbms.SigmoidBinaryRBM(n_visible, n_hidden)
  rbm = morb.base.RBM()
  rbm.v = units.BinaryUnits(rbm, name='v') # visibles
  rbm.h = units.BinaryUnits(rbm, name='h') # hiddens
  rbm.W = parameters.Convolutional2DParameters(rbm, [rbm.v, rbm.h], theano.shared(value=initial_W, name='W'), name='W', shape_info=shape_info)
  # one bias per map (so shared across width and height):
  rbm.bv = parameters.SharedBiasParameters(rbm, rbm.v, 3, 2, theano.shared(value=initial_bv, name='bv'), name='bv')
  rbm.bh = parameters.SharedBiasParameters(rbm, rbm.h, 3, 2, theano.shared(value=initial_bh, name='bh'), name='bh')

  initial_vmap = { rbm.v: T.tensor4('v') }

  # try to calculate weight updates using CD-1 stats
  print ">> Constructing contrastive divergence updaters..."
  s = stats.cd_stats(rbm, initial_vmap, visible_units=[rbm.v], hidden_units=[rbm.h], k=5, mean_field_for_stats=[rbm.v], mean_field_for_gibbs=[rbm.v])

  umap = {}
  for var in rbm.variables:
    pu =  var + 0.001 * updaters.CDUpdater(rbm, var, s)
    umap[var] = pu

  print ">> Compiling functions..."
  t = trainers.MinibatchTrainer(rbm, umap)
  m = monitors.reconstruction_mse(s, rbm.v)

  e_data = rbm.energy(s['data']).mean()
  e_model = rbm.energy(s['model']).mean()


  # train = t.compile_function(initial_vmap, mb_size=32, monitors=[m], name='train', mode=mode)
  train = t.compile_function(initial_vmap, mb_size=mb_size, monitors=[m, e_data, e_model], name='train', mode=mode)


  # TRAINING 

  epochs = epoch_cd
  print ">> Training for %d epochs..." % epochs



  for epoch in range(epochs):
    monitoring_data_train = [(cost, energy_data, energy_model) for cost, energy_data, energy_model in train({ rbm.v: train_set_x })]
    mses_train, edata_train_list, emodel_train_list = zip(*monitoring_data_train)
  
  
  #print rbm.W.var.get_value().shape
  lay1w = rbm.W.var.get_value()
  Wl = theano.shared(lay1w) 
  lay1bh = rbm.bh.var.get_value() 
  bhl = theano.shared(lay1bh)
  #print Wl.get_value().shape
  return [Wl, bhl]
 
class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), W=None, b=None, bmode='valid'):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input
        self.W = W
        self.b = b

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
                   
        # initialize weights with random weights
        if self.W is None:
          W_bound = numpy.sqrt(6. / (fan_in + fan_out))
          self.W = theano.shared(
             numpy.asarray(
                 rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                 dtype=theano.config.floatX
             ),
             borrow=True
           )

        # the bias is a 1D tensor -- one bias per output feature map
        if b is None:
          b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
          self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape,
            border_mode = bmode 
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


def evaluate_lenet5(learning_rate=2, n_epochs=100,
                    dataset='F:/MOUD/MOUD/jul14/x50_1/cktest/moud6.pkl.gz',
                    nkerns=[5, 5, 5, 5, 5, 5, 5, 5, 5], batch_size=50, dirn='iti', indexd=0):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    global layer0gW
    global layer1gW
    global layer1bgW
    global layer1cgW
    global layer1dgW
    global layer1egW
    global layer1fgW
    global layer1ggW
    global layer1hgW
    global layer2gW
    global layer3gW
    global layer0gb
    global layer1gb
    global layer1bgb
    global layer1cgb
    global layer1dgb
    global layer1egb
    global layer1fgb
    global layer1ggW
    global layer1hgW
    global layer2gb
    global layer3gb
    global all_test
    global batchm  
    global eval_print1
    global eval_print2
    global eval_print3
    global neuron
    global epoch_cd
    
    epoch_cd = 2
    neuron = 10
    batchm  = 50
    batch_size = batchm
    
    for nk in range(9):
        nkerns[nk]=neuron
      
    indk = 10;
    dirgtest = dirn;
     
    if indexd > indk:
    
        f = file(dirgtest+"/weights/layer0w.save",'rb')
        layer0gW = theano.shared(cPickle.load(f)) 
        f.close()
        
        f = file(dirgtest+"/weights/layer1w.save",'rb')
        layer1gW = theano.shared(cPickle.load(f))
        f.close()
    
        f = file(dirgtest+"/weights/layer2w.save",'rb')
        layer2gW = theano.shared(cPickle.load(f)) 
        f.close()
        
        f = file(dirgtest+"/weights/layer3w.save",'rb')
        layer3gW = theano.shared(cPickle.load(f)) 
        f.close()
        
        f = file(dirgtest+"/weights/layer0b.save",'rb')
        layer0gb = theano.shared(cPickle.load(f)) 
        f.close()
        
        f = file(dirgtest+"/weights/layer1b.save",'rb')
        layer1gb = theano.shared(cPickle.load(f)) 
        f.close()
    
        f = file(dirgtest+"/weights/layer2b.save",'rb')
        layer2gb = theano.shared(cPickle.load(f)) 
        f.close()
        
        f = file(dirgtest+"/weights/layer3b.save",'rb')
        layer3gb = theano.shared(cPickle.load(f)) 
        f.close()
    
    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2] 
     
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
                                                                                                                                                                                                                 
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    if indexd > indk:           
        n_epochs = 1;

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    im1x = 125
    im1y =  250
    poolx = 1
    pooly = 1
    
    layer0_input = x.reshape((batch_size, 1, im1x, im1y))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    nk1x=4
    nk1y=4#im1y
    Wl = None;
    bl = None;
    
    if indexd > indk:           
        Wl = layer0gW
        bl = layer0gb         
                                                                                       
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, im1x, im1y),
        filter_shape=(nkerns[0], 1, nk1x, nk1y),
        poolsize=(poolx, pooly),
        W = Wl,
        b = bl
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    im2x = (im1x-nk1x+1)/poolx
    im2y = (im1y-nk1y+1)/pooly
    #im2x = (im1x+nk1x-1)/poolx
    #im2y = (im1y+nk1y-1)/pooly
    nk2x=4
    nk2y=4#im2y
     
    Wl = None;
    bl = None;
    
    if indexd > indk:           
        Wl = layer1gW
        bl = layer1gb
             
    
    
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], im2x, im2y),
        filter_shape=(nkerns[1], nkerns[0], nk2x, nk2y),
        poolsize=(poolx, pooly),
        W = Wl,
        b = bl
    )

#    # Construct the third convolutional pooling layer
#    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
#    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
#    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    im2bx = (im2x-nk2x+1)/poolx
    im2by = (im2y-nk2y+1)/pooly
    #im2bx = (im2x+nk2x-1)/poolx
    #im2by = (im2y+nk2y-1)/pooly
    nk2bx=4
    nk2by=4#im2by
    Wl = None;
    bl = None;
    
   
    if indexd > indk:           
        Wl = layer1bgW
        bl = layer1bgb
             
    
    layer1b = LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], im2bx, im2by),
        filter_shape=(nkerns[2], nkerns[1], nk2bx, nk2by),
        poolsize=(poolx, pooly),
        W = Wl,
        b = bl
    )

    # Construct the fourth convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    im2cx = (im2bx-nk2bx+1)/poolx
    im2cy = (im2by-nk2by+1)/pooly
    nk2cx=4
    nk2cy=4#im2cy
    Wl = None;
    bl = None;
    
    if indexd > indk:           
        Wl = layer1cgW
        bl = layer1cgb
             
     
    
    layer1c = LeNetConvPoolLayer(
        rng,
        input=layer1b.output,
        image_shape=(batch_size, nkerns[2], im2cx, im2cy),
        filter_shape=(nkerns[3], nkerns[2], nk2cx, nk2cy),
        poolsize=(poolx, pooly),
        W = Wl,
        b = bl
    )
    
        # Construct the fifth convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    im2dx = (im2cx-nk2cx+1)/poolx
    im2dy = (im2cy-nk2cy+1)/pooly
    nk2dx=4
    nk2dy=4#im2dy
    Wl = None;
    bl = None;
    
    if indexd > indk:           
        Wl = layer1dgW
        bl = layer1dgb
             
    
    layer1d = LeNetConvPoolLayer(
        rng,
        input=layer1c.output,
        image_shape=(batch_size, nkerns[3], im2dx, im2dy),
        filter_shape=(nkerns[4], nkerns[3], nk2dx, nk2dy),
        poolsize=(poolx, pooly),
        W = Wl,
        b = bl
    )

    # Construct the sixth convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    im2ex = (im2dx-nk2dx+1)/poolx
    im2ey = (im2dy-nk2dy+1)/pooly
    nk2ex=4
    nk2ey=4#im2ey
    Wl = None;
    bl = None;
    
    if indexd > indk:           
        Wl = layer1egW
        bl = layer1egb
             
    
    layer1e = LeNetConvPoolLayer(
        rng,
        input=layer1d.output,
        image_shape=(batch_size, nkerns[4], im2ex, im2ey),
        filter_shape=(nkerns[5], nkerns[4], nk2ex, nk2ey),
        poolsize=(poolx, pooly),
        W = Wl,
        b = bl
    )
    
        # Construct the seven convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    im2fx = (im2ex-nk2ex+1)/poolx
    im2fy = (im2ey-nk2ey+1)/pooly
    nk2fx=4
    nk2fy=4#im2fy
    Wl = None;
    bl = None;
    
    if indexd > indk:           
        Wl = layer1fgW
        bl = layer1fgb
   
    
    layer1f = LeNetConvPoolLayer(
        rng,
        input=layer1e.output,
        image_shape=(batch_size, nkerns[5], im2fx, im2fy),
        filter_shape=(nkerns[6], nkerns[5], nk2fx, nk2fy),
        poolsize=(poolx, pooly),
        W = Wl,
        b = bl
    )
    
        # Construct the eight convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    im2gx = (im2fx-nk2fx+1)/poolx
    im2gy = (im2fy-nk2fy+1)/pooly
    nk2gx=4
    nk2gy=4#im2gy
    Wl = None;
    bl = None;
    
    if indexd > indk:           
        Wl = layer1ggW
        bl = layer1ggb

    
    layer1g = LeNetConvPoolLayer(
        rng,
        input=layer1f.output,
        image_shape=(batch_size, nkerns[6], im2gx, im2gy),
        filter_shape=(nkerns[7], nkerns[6], nk2gx, nk2gy),
        poolsize=(poolx, pooly),
        W = Wl,
        b = bl
    )
    
        # Construct the ninth convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    im2hx = (im2gx-nk2gx+1)/poolx
    im2hy = (im2gy-nk2gy+1)/pooly
    nk2hx=4
    nk2hy=4#im2hy
    Wl = None;
    bl = None;
    
    if indexd > indk:           
        Wl = layer1hgW
        bl = layer1hgb
    
    layer1h = LeNetConvPoolLayer(
        rng,
        input=layer1g.output,
        image_shape=(batch_size, nkerns[7], im2hx, im2hy),
        filter_shape=(nkerns[8], nkerns[7], nk2hx, nk2hy),
        poolsize=(poolx, pooly),
        W = Wl,
        b = bl
    )
    

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1h.output.flatten(2)
    im3x = (im2hx-nk2hx+1)/poolx
    im3y = (im2hy-nk2hy+1)/pooly
    Wl = None;
    bl = None;
    
    # construct a fully-connected sigmoidal layer
    if indexd > indk:           
        Wl = layer2gW
        bl = layer2gb
    
    L1_reg = 0.01    
    L2_reg = 0.01
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[4] * im3x * im3y,
        n_out=20,
        activation=T.tanh,
        W = Wl,
        b = bl
    )
    
    layer2.L1 = L1_reg
    layer2.L2_sqr = L2_reg
    
    if indexd > indk:           
        Wl = layer3gW
        bl = layer3gb
    
                  
    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=20, n_out=2, W = Wl, b = bl)

    # the cost we minimize during training is the NLL of the model
   
    cost = layer3.negative_log_likelihood(y)
    
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1h.params + layer1g.params + layer1f.params + layer1e.params + layer1d.params + layer1c.params + layer1b.params + layer1.params + layer0.params


    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
                               
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        learning_rate = 0.99*learning_rate
        if epoch == 1:
             print layer0.W.get_value().shape
             print layer0.b.get_value().shape
             
             eval_set_x = test_set_x;
             eval_shape = train_set_x.get_value(borrow=True).shape; 
             eval_layer2 = theano.function([index], layer0_input,
                givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})
             eval_print1 = [
                   eval_layer2(i)
                   for i in xrange(n_test_batches)
                 ]  
                                                                                                                                                                                                                                                          
             eval_set_x = train_set_x;
             
             eval_layer2 = theano.function([index], layer0_input,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print2 = [
                   eval_layer2(i)
                   for i in xrange(n_train_batches)
                 ]  
    
             eval_set_x = valid_set_x;
       
             eval_layer2 = theano.function([index], layer0_input,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print3 = [
                   eval_layer2(i)
                   for i in xrange(n_valid_batches)
                 ]
                        
             Wl1, bl1 = morbrun1(nk1x,nk1y,im1x,im1y)
             layer0 = LeNetConvPoolLayer(
                  rng,
                  input=layer0_input,
                  image_shape=(batch_size, 1 , im1x, im1y),
                  filter_shape=(nkerns[0], 1 , nk1x, nk1y),
                  poolsize=(poolx, pooly),
                  W = Wl1,
                  b = bl1
             )     
                
        if epoch == 2:
            
             print layer1.W.get_value().shape
             print layer1.b.get_value().shape
             
             eval_set_x = test_set_x;
             eval_shape = train_set_x.get_value(borrow=True).shape; 
             eval_layer2 = theano.function([index], layer0.output,
                givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})
             eval_print1 = [
                   eval_layer2(i)
                   for i in xrange(n_test_batches)
                 ]  
                                                                                                                                                                                                                                                          
             eval_set_x = train_set_x;
             
             eval_layer2 = theano.function([index], layer0.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print2 = [
                   eval_layer2(i)
                   for i in xrange(n_train_batches)
                 ]  
    
             eval_set_x = valid_set_x;
       
             eval_layer2 = theano.function([index], layer0.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print3 = [
                   eval_layer2(i)
                   for i in xrange(n_valid_batches)
                 ]  
                 
             Wl2, bl2 = morbrun1(nk2x,nk2y,im2x,im2y,neuron)
             
             layer1 = LeNetConvPoolLayer(
                  rng,
                  input=layer0.output,
                  image_shape=(batch_size, nkerns[0], im2x, im2y),
                  filter_shape=(nkerns[1], nkerns[0], nk2x, nk2y),
                  poolsize=(poolx, pooly),
                  W = Wl2,
                  b = bl2
             )
             layer0 = LeNetConvPoolLayer(
                  rng,
                  input=layer0_input,
                  image_shape=(batch_size, 1 , im1x, im1y),
                  filter_shape=(nkerns[0], 1 , nk1x, nk1y),
                  poolsize=(poolx, pooly),
                  W = Wl1,
                  b = bl1
             )  
        
        
        if epoch == 3:
             print layer1b.W.get_value().shape
             eval_set_x = test_set_x;
             eval_shape = train_set_x.get_value(borrow=True).shape; 
             eval_layer2 = theano.function([index], layer1.output,
                givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})
             eval_print1 = [
                   eval_layer2(i)
                   for i in xrange(n_test_batches)
                 ]  
                                                                                                                                                                                                                                                          
             eval_set_x = train_set_x;
             
             eval_layer2 = theano.function([index], layer1.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print2 = [
                   eval_layer2(i)
                   for i in xrange(n_train_batches)
                 ]  
    
             eval_set_x = valid_set_x;
       
             eval_layer2 = theano.function([index], layer1.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print3 = [
                   eval_layer2(i)
                   for i in xrange(n_valid_batches)
                 ]  
                 
             Wl3, bl3 = morbrun1(nk2bx,nk2by,im2bx,im2by,neuron)
             
             layer1b = LeNetConvPoolLayer(
                  rng,
                  input=layer1.output,
                  image_shape=(batch_size, nkerns[1], im2bx, im2by),
                  filter_shape=(nkerns[2], nkerns[1], nk2bx, nk2by),
                  poolsize=(poolx, pooly),
                  W = Wl3,
                  b = bl3
             )
             layer1 = LeNetConvPoolLayer(
                  rng,
                  input=layer0.output,
                  image_shape=(batch_size, nkerns[0], im2x, im2y),
                  filter_shape=(nkerns[1], nkerns[0], nk2x, nk2y),
                  poolsize=(poolx, pooly),
                  W = Wl2,
                  b = bl2
             )
             layer0 = LeNetConvPoolLayer(
                  rng,
                  input=layer0_input,
                  image_shape=(batch_size, 1 , im1x, im1y),
                  filter_shape=(nkerns[0], 1 , nk1x, nk1y),
                  poolsize=(poolx, pooly),
                  W = Wl1,
                  b = bl1
             )  
             
        if epoch == 4:
             print layer1c.W.get_value().shape
             eval_set_x = test_set_x;
             eval_shape = train_set_x.get_value(borrow=True).shape; 
             eval_layer2 = theano.function([index], layer1b.output,
                givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})
             eval_print1 = [
                   eval_layer2(i)
                   for i in xrange(n_test_batches)
                 ]  
                                                                                                                                                                                                                                                          
             eval_set_x = train_set_x;
             
             eval_layer2 = theano.function([index], layer1b.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print2 = [
                   eval_layer2(i)
                   for i in xrange(n_train_batches)
                 ]  
    
             eval_set_x = valid_set_x;
       
             eval_layer2 = theano.function([index], layer1b.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print3 = [
                   eval_layer2(i)
                   for i in xrange(n_valid_batches)
                 ]  
                 
             Wl4, bl4 = morbrun1(nk2cx,nk2cy,im2cx,im2cy,neuron)    
             
             layer1c = LeNetConvPoolLayer(
                  rng,
                  input=layer1b.output,
                  image_shape=(batch_size, nkerns[2], im2cx, im2cy),
                  filter_shape=(nkerns[3], nkerns[2], nk2cx, nk2cy),
                  poolsize=(poolx, pooly),
                  W = Wl4,
                  b = bl4
             )
             layer1b = LeNetConvPoolLayer(
                  rng,
                  input=layer1.output,
                  image_shape=(batch_size, nkerns[1], im2bx, im2by),
                  filter_shape=(nkerns[2], nkerns[1], nk2bx, nk2by),
                  poolsize=(poolx, pooly),
                  W = Wl3,
                  b = bl3
             )
             layer1 = LeNetConvPoolLayer(
                  rng,
                  input=layer0.output,
                  image_shape=(batch_size, nkerns[0], im2x, im2y),
                  filter_shape=(nkerns[1], nkerns[0], nk2x, nk2y),
                  poolsize=(poolx, pooly),
                  W = Wl2,
                  b = bl2
             )
             layer0 = LeNetConvPoolLayer(
                  rng,
                  input=layer0_input,
                  image_shape=(batch_size, 1 , im1x, im1y),
                  filter_shape=(nkerns[0], 1 , nk1x, nk1y),
                  poolsize=(poolx, pooly),
                  W = Wl1,
                  b = bl1
             )  
             
        if epoch == 5:
             print layer1d.W.get_value().shape
             eval_set_x = test_set_x;
             eval_shape = train_set_x.get_value(borrow=True).shape; 
             eval_layer2 = theano.function([index], layer1c.output,
                givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})
             eval_print1 = [
                   eval_layer2(i)
                   for i in xrange(n_test_batches)
                 ]  
                                                                                                                                                                                                                                                          
             eval_set_x = train_set_x;
             
             eval_layer2 = theano.function([index], layer1c.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print2 = [
                   eval_layer2(i)
                   for i in xrange(n_train_batches)
                 ]  
    
             eval_set_x = valid_set_x;
       
             eval_layer2 = theano.function([index], layer1c.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print3 = [
                   eval_layer2(i)
                   for i in xrange(n_valid_batches)
                 ]  
                 
             
             Wl5, bl5 = morbrun1(nk2dx,nk2dy,im2dx,im2dy,neuron)
             
             layer1d = LeNetConvPoolLayer(
                  rng,
                  input=layer1c.output,
                  image_shape=(batch_size, nkerns[3], im2dx, im2dy),
                  filter_shape=(nkerns[4], nkerns[3], nk2dx, nk2dy),
                  poolsize=(poolx, pooly),
                  W = Wl5,
                  b = bl5
             )  
             layer1c = LeNetConvPoolLayer(
                  rng,
                  input=layer1b.output,
                  image_shape=(batch_size, nkerns[2], im2cx, im2cy),
                  filter_shape=(nkerns[3], nkerns[2], nk2cx, nk2cy),
                  poolsize=(poolx, pooly),
                  W = Wl4,
                  b = bl4
             )
             layer1b = LeNetConvPoolLayer(
                  rng,
                  input=layer1.output,
                  image_shape=(batch_size, nkerns[1], im2bx, im2by),
                  filter_shape=(nkerns[2], nkerns[1], nk2bx, nk2by),
                  poolsize=(poolx, pooly),
                  W = Wl3,
                  b = bl3
             )
             layer1 = LeNetConvPoolLayer(
                  rng,
                  input=layer0.output,
                  image_shape=(batch_size, nkerns[0], im2x, im2y),
                  filter_shape=(nkerns[1], nkerns[0], nk2x, nk2y),
                  poolsize=(poolx, pooly),
                  W = Wl2,
                  b = bl2
             )
             layer0 = LeNetConvPoolLayer(
                  rng,
                  input=layer0_input,
                  image_shape=(batch_size, 1 , im1x, im1y),
                  filter_shape=(nkerns[0], 1 , nk1x, nk1y),
                  poolsize=(poolx, pooly),
                  W = Wl1,
                  b = bl1
             )  
                     
        if epoch == 6:
             print layer1e.W.get_value().shape
             eval_set_x = test_set_x;
             eval_shape = train_set_x.get_value(borrow=True).shape; 
             eval_layer2 = theano.function([index], layer1d.output,
                givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})
             eval_print1 = [
                   eval_layer2(i)
                   for i in xrange(n_test_batches)
                 ]  
                                                                                                                                                                                                                                                          
             eval_set_x = train_set_x;
             
             eval_layer2 = theano.function([index], layer1d.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print2 = [
                   eval_layer2(i)
                   for i in xrange(n_train_batches)
                 ]  
    
             eval_set_x = valid_set_x;
       
             eval_layer2 = theano.function([index], layer1d.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print3 = [
                   eval_layer2(i)
                   for i in xrange(n_valid_batches)
                 ]  
                 
             
             Wl6,bl6 = morbrun1(nk2ex,nk2ey,im2ex,im2ey,neuron)
             
             layer1e = LeNetConvPoolLayer(
                  rng,
                  input=layer1d.output,
                  image_shape=(batch_size, nkerns[4], im2ex, im2ey),
                  filter_shape=(nkerns[5], nkerns[4], nk2ex, nk2ey),
                  poolsize=(poolx, pooly),
                  W = Wl6,
                  b = bl6
             ) 
             layer1d = LeNetConvPoolLayer(
                  rng,
                  input=layer1c.output,
                  image_shape=(batch_size, nkerns[3], im2dx, im2dy),
                  filter_shape=(nkerns[4], nkerns[3], nk2dx, nk2dy),
                  poolsize=(poolx, pooly),
                  W = Wl5,
                  b = bl5
             )  
             layer1c = LeNetConvPoolLayer(
                  rng,
                  input=layer1b.output,
                  image_shape=(batch_size, nkerns[2], im2cx, im2cy),
                  filter_shape=(nkerns[3], nkerns[2], nk2cx, nk2cy),
                  poolsize=(poolx, pooly),
                  W = Wl4,
                  b = bl4
             )
             layer1b = LeNetConvPoolLayer(
                  rng,
                  input=layer1.output,
                  image_shape=(batch_size, nkerns[1], im2bx, im2by),
                  filter_shape=(nkerns[2], nkerns[1], nk2bx, nk2by),
                  poolsize=(poolx, pooly),
                  W = Wl3,
                  b = bl3
             )
             layer1 = LeNetConvPoolLayer(
                  rng,
                  input=layer0.output,
                  image_shape=(batch_size, nkerns[0], im2x, im2y),
                  filter_shape=(nkerns[1], nkerns[0], nk2x, nk2y),
                  poolsize=(poolx, pooly),
                  W = Wl2,
                  b = bl2
             )
             layer0 = LeNetConvPoolLayer(
                  rng,
                  input=layer0_input,
                  image_shape=(batch_size, 1 , im1x, im1y),
                  filter_shape=(nkerns[0], 1 , nk1x, nk1y),
                  poolsize=(poolx, pooly),
                  W = Wl1,
                  b = bl1
             )              
        if epoch == 7:
             print layer1f.W.get_value().shape
             eval_set_x = test_set_x;
             eval_shape = train_set_x.get_value(borrow=True).shape; 
             eval_layer2 = theano.function([index], layer1e.output,
                givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})
             eval_print1 = [
                   eval_layer2(i)
                   for i in xrange(n_test_batches)
                 ]  
                                                                                                                                                                                                                                                          
             eval_set_x = train_set_x;
             
             eval_layer2 = theano.function([index], layer1e.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print2 = [
                   eval_layer2(i)
                   for i in xrange(n_train_batches)
                 ]  
    
             eval_set_x = valid_set_x;
       
             eval_layer2 = theano.function([index], layer1e.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print3 = [
                   eval_layer2(i)
                   for i in xrange(n_valid_batches)
                 ]  
                 
             
             Wl7,bl7 = morbrun1(nk2fx,nk2fy,im2fx,im2fy,neuron)
             
             layer1f = LeNetConvPoolLayer(
                  rng,
                  input=layer1e.output,
                  image_shape=(batch_size, nkerns[5], im2fx, im2fy),
                  filter_shape=(nkerns[6], nkerns[5], nk2fx, nk2fy),
                  poolsize=(poolx, pooly),
                  W = Wl7,
                  b = bl7
             )
             layer1e = LeNetConvPoolLayer(
                  rng,
                  input=layer1d.output,
                  image_shape=(batch_size, nkerns[4], im2ex, im2ey),
                  filter_shape=(nkerns[5], nkerns[4], nk2ex, nk2ey),
                  poolsize=(poolx, pooly),
                  W = Wl6,
                  b = bl6
             ) 
             layer1d = LeNetConvPoolLayer(
                  rng,
                  input=layer1c.output,
                  image_shape=(batch_size, nkerns[3], im2dx, im2dy),
                  filter_shape=(nkerns[4], nkerns[3], nk2dx, nk2dy),
                  poolsize=(poolx, pooly),
                  W = Wl5,
                  b = bl5
             )  
             layer1c = LeNetConvPoolLayer(
                  rng,
                  input=layer1b.output,
                  image_shape=(batch_size, nkerns[2], im2cx, im2cy),
                  filter_shape=(nkerns[3], nkerns[2], nk2cx, nk2cy),
                  poolsize=(poolx, pooly),
                  W = Wl4,
                  b = bl4
             )
             layer1b = LeNetConvPoolLayer(
                  rng,
                  input=layer1.output,
                  image_shape=(batch_size, nkerns[1], im2bx, im2by),
                  filter_shape=(nkerns[2], nkerns[1], nk2bx, nk2by),
                  poolsize=(poolx, pooly),
                  W = Wl3,
                  b = bl3
             )
             layer1 = LeNetConvPoolLayer(
                  rng,
                  input=layer0.output,
                  image_shape=(batch_size, nkerns[0], im2x, im2y),
                  filter_shape=(nkerns[1], nkerns[0], nk2x, nk2y),
                  poolsize=(poolx, pooly),
                  W = Wl2,
                  b = bl2
             )
             layer0 = LeNetConvPoolLayer(
                  rng,
                  input=layer0_input,
                  image_shape=(batch_size, 1 , im1x, im1y),
                  filter_shape=(nkerns[0], 1 , nk1x, nk1y),
                  poolsize=(poolx, pooly),
                  W = Wl1,
                  b = bl1
             )              
        if epoch == 8:
             print layer1g.W.get_value().shape
             eval_set_x = test_set_x;
             eval_shape = train_set_x.get_value(borrow=True).shape; 
             eval_layer2 = theano.function([index], layer1f.output,
                givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})
             eval_print1 = [
                   eval_layer2(i)
                   for i in xrange(n_test_batches)
                 ]  
                                                                                                                                                                                                                                                          
             eval_set_x = train_set_x;
             
             eval_layer2 = theano.function([index], layer1f.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print2 = [
                   eval_layer2(i)
                   for i in xrange(n_train_batches)
                 ]  
    
             eval_set_x = valid_set_x;
       
             eval_layer2 = theano.function([index], layer1f.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print3 = [
                   eval_layer2(i)
                   for i in xrange(n_valid_batches)
                 ]  
                 
             
             Wl, bl = morbrun1(nk2gx,nk2gy,im2gx,im2gy,neuron)
             
             layer1g = LeNetConvPoolLayer(
                  rng,
                  input=layer1f.output,
                  image_shape=(batch_size, nkerns[6], im2gx, im2gy),
                  filter_shape=(nkerns[7], nkerns[6], nk2gx, nk2gy),
                  poolsize=(poolx, pooly),
                  W = Wl,
                  b = bl
             )                 
        if epoch == 9: 
             print layer1h.W.get_value().shape
             eval_set_x = test_set_x;
             eval_shape = train_set_x.get_value(borrow=True).shape; 
             eval_layer2 = theano.function([index], layer1g.output,
                givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})
             eval_print1 = [
                   eval_layer2(i)
                   for i in xrange(n_test_batches)
                 ]  
                                                                                                                                                                                                                                                          
             eval_set_x = train_set_x;
             
             eval_layer2 = theano.function([index], layer1g.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print2 = [
                   eval_layer2(i)
                   for i in xrange(n_train_batches)
                 ]  
    
             eval_set_x = valid_set_x;
       
             eval_layer2 = theano.function([index], layer1g.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

             eval_print3 = [
                   eval_layer2(i)
                   for i in xrange(n_valid_batches)
                 ]  
                 
            
             Wl, bl = morbrun1(nk2hx,nk2hy,im2hx,im2hy,neuron)
             
             layer1h = LeNetConvPoolLayer(
                  rng,
                  input=layer1g.output,
                  image_shape=(batch_size, nkerns[7], im2hx, im2hy),
                  filter_shape=(nkerns[8], nkerns[7], nk2hx, nk2hy),
                  poolsize=(poolx, pooly),
                  W = Wl,
                  b = bl
             )  
                                                                      
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    all_test += test_score
    print str(all_test)+' '+str(indexd) 
    
    with open(dirgtest+"/run2/cv_score.txt", "a") as myfile:
          myfile.write(str(test_score)+"\n")
         
    #print layer0gW[0,0,:,:]        
                                          
    # save weights in global variables
    if indexd == indk:
      layer0gW = layer0.W;  
      layer1gW = layer1.W;
      layer2gW = layer2.W;
      layer3gW = layer3.W;                                    
      layer0gb = layer0.b;  
      layer1gb = layer1.b;
      layer2gb = layer2.b;
      layer3gb = layer3.b;
                                                           
    # print the kernels   
                                                                                                                                                                                                                                                                                                              
     # print layer0.W.get_value()                       
#    hea = layer0.W.get_value(True)
#    with open("layer0_w.txt", "w") as myfile:
#              hea.tofile(myfile,sep=" ",format="%s")     
#              myfile.write('\n')
#    myfile.close()   
     
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    if indexd == indexd: 
        
        eval_set_x = test_set_x;
        eval_shape = train_set_x.get_value(borrow=True).shape; 
    
        eval_layer2 = theano.function([index], layer2.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

    

        eval_print = [
                   eval_layer2(i)
                   for i in xrange(n_test_batches)
                 ]  
                 
                 
        numpy.set_printoptions(threshold=sys.maxint)         
                                                                                                                       
        with open(dirgtest+"/run2/layer0_vid_test"+str(indexd)+".csv", "w") as myfile:
              writer1 = csv.writer(myfile);
              for item in eval_print:
                    writer1.writerow(item);       
              myfile.close()                         
                                                      
                                                                             
                                                                                                                           
        eval_set_x = train_set_x;
        eval_shape = train_set_x.get_value(borrow=True).shape; 
    
        eval_layer2 = theano.function([index], layer2.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

        eval_print = [
                   eval_layer2(i)
                   for i in xrange(n_train_batches)
                 ]  
         
                         
        numpy.set_printoptions(threshold=sys.maxint)                                                                                                                        
        with open(dirgtest+"/run2/layer0_vid_train"+str(indexd)+".csv", "w") as myfile:
              writer1 = csv.writer(myfile);
              for item in eval_print:
                    writer1.writerow(item);      
              myfile.close()

        eval_set_x = valid_set_x;
        eval_shape = train_set_x.get_value(borrow=True).shape; 
    
        eval_layer2 = theano.function([index], layer2.output,
             givens={
                x: eval_set_x[index * batch_size: (index + 1) * batch_size]})

        eval_print = [
                   eval_layer2(i)
                   for i in xrange(n_valid_batches)
                 ]  
                 
        
                 
        numpy.set_printoptions(threshold=sys.maxint)                                                                                                                        
        with open(dirgtest+"/run2/layer0_vid_val"+str(indexd)+".csv", "w") as myfile:
              writer1 = csv.writer(myfile); 
              for item in eval_print:
                   writer1.writerow(item);       
              myfile.close()
   
    if indexd == indexd:            
                                    
        f = file(dirgtest+"/weights/layer0w_"+str(indexd)+".save", 'wb')
        cPickle.dump(layer0.W.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()  
        
        f = file(dirgtest+"/weights/layer1w_"+str(indexd)+".save", 'wb')
        cPickle.dump(layer1.W.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        
        f = file(dirgtest+"/weights/layer2w_"+str(indexd)+".save", 'wb')
        cPickle.dump(layer2.W.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        
        #print layer2.W[0].get_value()
        
        f = file(dirgtest+"/weights/layer3w_"+str(indexd)+".save", 'wb')
        cPickle.dump(layer3.W.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        
        f = file(dirgtest+"/weights/layer0b_"+str(indexd)+".save", 'wb')
        cPickle.dump(layer0.b.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        
        f = file(dirgtest+"/weights/layer1b_"+str(indexd)+".save", 'wb')
        cPickle.dump(layer1.b.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        
        f = file(dirgtest+"/weights/layer2b_"+str(indexd)+".save", 'wb')
        cPickle.dump(layer2.b.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        
        f = file(dirgtest+"/weights/layer3b_"+str(indexd)+".save", 'wb')
        cPickle.dump(layer3.b.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()

if __name__ == '__main__':
    dirgtest = '.'
    
    try:
       os.remove(dirgtest+"/run2/cv_score.txt")
    except OSError:
        pass
    
    
    for i in range(1):
          k = i+5
       #   nameg = dirgtest+'/../fold/jp_en_dvd'+str(k)+'.pkl.gz'       
       #   nameg = '../moud'+str(k)+'b.pkl.gz'
          nameg = '../cnninput/x50_'+str(k+1)+'/moudb'+str(k+1)+'cv.pkl.gz'
          evaluate_lenet5(dataset=nameg, n_epochs = 10, dirn=dirgtest, indexd=k)
        
def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
