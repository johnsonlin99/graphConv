# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:39:14 2019

@author: Johnson
"""

import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '../')
import utils
import os

# tells which GPU to use.
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import matplotlib.pyplot as plt
import seaborn as sns
import graphConv as gc

# =============================================================================
# Defining model
# =============================================================================
# 3 layers of graphconv + 2 layers of dense.
class PPiConv:
    
    def __init__(self, ppi, gconvlayer, outsize=2, batchsize=64):
        # Parameters
        self.ppi = ppi
        self.batchsize = batchsize
        self.outsize = outsize
        self.genesize = 16559
        self._lambda = 0.0
        
        # Make a graph
        tf.reset_default_graph()
        self.built = False
        self.sesh = tf.Session()
        self.ops = self.build(gconvlayer)
        self.sesh.run(tf.global_variables_initializer())
        
        # Stats to keep track
        self.e = 0
        self.loss = {"train":[], "val":[]}
        
    def build(self, gconvlayer):
        # Placeholders for input and dropout probs.
        if self.built:
            return -1
        else:
            self.built = True
            
        #Get DAD matrix (1st order approx. of Graph laplacian)
        #This implements eq7 and 8 of the paper.
        #We are using sparse tensor class
        DAD = tf.SparseTensor(indices=self.ppi[0], values=tf.constant(self.ppi[1], dtype=tf.float32),\
                              dense_shape=[self.genesize, self.genesize])
        
        #Get data matrix to convolve
        x = tf.placeholder(shape=[self.batchsize, self.genesize, 1], dtype=tf.float32)
        y = tf.placeholder(shape=[self.batchsize, self.outsize], dtype=tf.float32)
        keep_probability = tf.placeholder_with_default(1.0, shape=[])
        
        # Three layers of convolution 1->5->5->5
        layer1 = gconvlayer(x, 5, DAD)
        layer1 = tf.nn.dropout(layer1, keep_prob=keep_probability)
        layer2 = gconvlayer(layer1, 5, DAD)
        layer2 = tf.nn.dropout(layer2, keep_prob=keep_probability)
        layer3 = gconvlayer(layer2, 2, DAD)
        layer3 = tf.nn.dropout(layer3, keep_prob=keep_probability)

        
        # Flatten inputs
        flattened = tf.layers.flatten(layer3)
        dense1 = tf.contrib.layers.fully_connected(flattened, 526, activation_fn=tf.nn.relu)
        dense1 = tf.nn.dropout(dense1, keep_prob=keep_probability)
        dense2 = tf.contrib.layers.fully_connected(dense1, 64, activation_fn=tf.nn.relu)
        
        # Prediction
        out = tf.contrib.layers.fully_connected(dense2, self.outsize, activation_fn=tf.identity)
        preds = tf.nn.softmax(out)
        
        # Define loss
        with tf.name_scope("l2_regularization"):
            regularizers = [tf.nn.l2_loss(v) for v in tf.trainable_variables() if "weights" in v.name]
            l2_reg = self._lambda * tf.add_n(regularizers)
            
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
            
        with tf.name_scope("Adam_optimizer"):
            optimizer = tf.train.AdamOptimizer()
            tvars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(loss+l2_reg, tvars)
            clipped = [(tf.clip_by_value(grad, -5, 5), tvar) for grad, tvar in grads_and_vars]
            train = optimizer.apply_gradients(clipped, name="minimize_cost")
        
        # Exporting out the operaions as dictionary
        return dict(
            x = x,
            y = y,
            keep_prob = keep_probability,
            train = train,
            cost = loss+l2_reg,
            preds = preds
        )
    
    def train(self, train, valid, epochs):
        # Defining the number of batches per epoch
        batch_num = int(np.ceil(train.n*1.0/train.batchsize))
        
        e = 0
        start_e = self.e
        while e < epochs:
            # Train
            total = []
            for i in range(batch_num):
                #Training happens here.
                batch = train.next()
                feed_dict = {self.ops["x"]: np.expand_dims(batch[0], -1),\
                             self.ops["y"]: batch[1]}
                
                ops_to_run = [self.ops["preds"],\
                              self.ops["cost"],\
                              self.ops["train"]]
                
                prediction, cost, _ = self.sesh.run(ops_to_run, feed_dict)
                total.append(cost)
                sys.stdout.write("\rEpoch: [%2d/%2d], Batch: [%2d/%2d], loss: %.2f"
                                 %(self.e, start_e+epochs, i, batch_num, cost))
            self.loss["train"].append(np.mean(total))
                
            # Validate
            total = []
            corrects = 0
            for i in range(0, valid.n, self.batchsize):
                feed_dict = {self.ops["x"]: np.expand_dims(valid.X[i:i+self.batchsize], -1),\
                            self.ops["y"]: valid.Y[i:i+self.batchsize]}
                
                ops_to_run = [self.ops["preds"],\
                              self.ops["cost"]]
                prediction, cost = self.sesh.run(ops_to_run, feed_dict)
                total.append(cost)
                corrects += np.sum((np.argmax(valid.Y[i:i+self.batchsize], axis=1) - np.argmax(prediction, axis=1)) == 0)
            
            self.loss["val"].append(np.mean(total))
            print(" valloss: %.2f"%(np.mean(total)),\
            " valacc: %.2f (%2d/%2d)"%(corrects*1.0/valid.n, corrects, valid.n))
            
            self.e+=1
            e+= 1
        return self.sesh.run([v for v in tf.trainable_variables() if "gweights" in v.name or "gbias" in v.name])
    
    def save(self, folder):
        saver = tf.train.Saver(tf.all_variables())
        os.system("mkdir "+folder)
        saver.save(self.sesh, folder+"/model.ckpt")
        
    def load(self, folder):
        saver = tf.train.Saver(tf.all_variables())
        saver.restore(self.sesh, folder+"/model.ckpt")

    # Encode examples
    def predict(self, x):
        feed_dict = {self.ops["x"]: x}
        return self.sesh.run(self.ops["preds"], feed_dict=feed_dict)

# Simply 3 layer dense net.
class Dense:
    
    def __init__(self, outsize=2, batchsize=64):
        # Parameters
        self.batchsize = batchsize
        self.outsize = outsize
        self.genesize = 16559
        self._lambda = 0.0
        
        # Make a graph
        tf.reset_default_graph()
        self.built = False
        self.sesh = tf.Session()
        self.ops = self.build()
        self.sesh.run(tf.global_variables_initializer())
        
        # Stats to keep track
        self.e = 0
        self.loss = {"train":[], "val":[]}
        
    def build(self):
        # Placeholders for input and dropout probs.
        if self.built:
            return -1
        else:
            self.built = True
        
        #Get data matrix to convolve
        x = tf.placeholder(shape=[self.batchsize, self.genesize, 1], dtype=tf.float32)
        y = tf.placeholder(shape=[self.batchsize, self.outsize], dtype=tf.float32)
        keep_probability = tf.placeholder_with_default(1.0, shape=[])
        isTraining = tf.placeholder_with_default(False, shape=[])
        
        # Flatten inputs
        flattened = tf.layers.flatten(x)
        #flattened = tf.contrib.layers.batch_norm(flattened, is_training=isTraining)

        dense1 = tf.contrib.layers.fully_connected(flattened, 1024, activation_fn=tf.nn.relu)
        dense1 = tf.nn.dropout(dense1, keep_prob=keep_probability)
        dense2 = tf.contrib.layers.fully_connected(dense1, 512, activation_fn=tf.nn.relu)
        dense2 = tf.nn.dropout(dense2, keep_prob=keep_probability)
        dense3 = tf.contrib.layers.fully_connected(dense2, 64, activation_fn=tf.nn.relu)
        dense3 = tf.nn.dropout(dense3, keep_prob=keep_probability)
        
        # Prediction
        out = tf.contrib.layers.fully_connected(dense3, self.outsize, activation_fn=tf.identity)
        preds = tf.nn.softmax(out)
        
        # Define loss
        with tf.name_scope("l2_regularization"):
            regularizers = [tf.nn.l2_loss(v) for v in tf.trainable_variables() if "weights" in v.name]
            l2_reg = self._lambda * tf.add_n(regularizers)
            
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=out))
            
        with tf.name_scope("Adam_optimizer"):
            optimizer = tf.train.AdamOptimizer()
            tvars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(loss+l2_reg, tvars)
            clipped = [(tf.clip_by_value(grad, -5, 5), tvar) for grad, tvar in grads_and_vars]
            train = optimizer.apply_gradients(clipped, name="minimize_cost")
        
        # Exporting out the operaions as dictionary
        return dict(
            x = x,
            y = y,
            keep_prob = keep_probability,
            isTraining = isTraining,
            train = train,
            cost = loss+l2_reg,
            preds = preds
        )
    
    def train(self, train, valid, epochs, dropout=1.0):
        # Defining the number of batches per epoch
        batch_num = int(np.ceil(train.n*1.0/train.batchsize))
        
        e = 0
        start_e = self.e
        while e < epochs:
            # Train
            total = []
            for i in range(batch_num):
                #Training happens here.
                batch = train.next()
                feed_dict = {self.ops["x"]: np.expand_dims(batch[0], -1),\
                             self.ops["y"]: batch[1],\
                             self.ops["keep_prob"]: dropout,\
                             self.ops["isTraining"]: True}
                
                ops_to_run = [self.ops["preds"],\
                              self.ops["cost"],\
                              self.ops["train"]]
                
                prediction, cost, _ = self.sesh.run(ops_to_run, feed_dict)
                total.append(cost)
                sys.stdout.write("\rEpoch: [%2d/%2d], Batch: [%2d/%2d], loss: %.2f"
                                 %(self.e, start_e+epochs, i, batch_num, cost))
            self.loss["train"].append(np.mean(total))
                
            # Validate
            total = []
            corrects = 0
            for i in range(0, valid.n, self.batchsize):
                feed_dict = {self.ops["x"]: np.expand_dims(valid.X[i:i+self.batchsize], -1),\
                            self.ops["y"]: valid.Y[i:i+self.batchsize]}
                
                ops_to_run = [self.ops["preds"],\
                              self.ops["cost"]]
                prediction, cost = self.sesh.run(ops_to_run, feed_dict)
                total.append(cost)
                corrects += np.sum((np.argmax(valid.Y[i:i+self.batchsize], axis=1) - np.argmax(prediction, axis=1)) == 0)
            
            self.loss["val"].append(np.mean(total))
            print(" valloss: %.2f"%(np.mean(total)),\
            " valacc: %.2f (%2d/%2d)"%(corrects*1.0/valid.n, corrects, valid.n))
            
            self.e+=1
            e+= 1
        return self.sesh.run([v for v in tf.trainable_variables() if "gweights" in v.name or "gbias" in v.name])
    
    def save(self, folder):
        saver = tf.train.Saver(tf.all_variables())
        os.system("mkdir "+folder)
        saver.save(self.sesh, folder+"/model.ckpt")
        
    def load(self, folder):
        saver = tf.train.Saver(tf.all_variables())
        saver.restore(self.sesh, folder+"/model.ckpt")

    # Encode examples
    def predict(self, x):
        feed_dict = {self.ops["x"]: x}
        return self.sesh.run(self.ops["preds"], feed_dict=feed_dict)

# =============================================================================
# Data loading
# =============================================================================
train_df = utils.datafeeder2(np.load("trainX.npy"),\
                            np.load("trainY.npy"))
valid_df = utils.datafeeder2(np.load("validX.npy"),\
                            np.load("validY.npy"))
test_df = utils.datafeeder2(np.load("testX.npy"),\
                           np.load("testY.npy"))

# =============================================================================
# PPI network loading and D^(-1/2)AD^(-1/2) matrix calculation
# =============================================================================
#Real ppi
ppi_matrix = np.load("ppi2.npy")
nom_adj_matrix = utils.preprocess_adj(ppi_matrix)
nom_adj_matrix2 = utils.preprocess_adj2(ppi_matrix)

# Fake ppi
fake_ppi = utils.ransomize_ppi(ppi_matrix)
nom_fake = utils.preprocess_adj(fake_ppi)
nom_fake2 = utils.preprocess_adj2(fake_ppi)

# No interaction at all
nom_noitx = utils.preprocess_adj(np.zeros((16559,16559)))
nom_noitx2 = utils.preprocess_adj2(np.zeros((16559,16559)))

# =============================================================================
# Training models
# =============================================================================
model1 = PPiConv(nom_adj_matrix, gc.convolutionGraph)
model1.train(train_df, valid_df, 20)
