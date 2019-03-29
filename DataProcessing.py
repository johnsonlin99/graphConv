# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:27:23 2019

@author: Johnson
"""

import numpy as np
import scipy.sparse as sp
import pandas as pd
import sklearn.preprocessing as prepro
import sys
import os

import sys
sys.path.insert(0, '../')
import utils


# =============================================================================
# Loading in RNAseq data from TCGA
# =============================================================================
# We first start by loading in rawdata obtained from Ayse. Ask her about what kind of batch correction happened.
# This is a feature by sample+1 2d matrix. +1 is just a label colmun
gene_profile = pd.read_csv("gene expression profile.txt", sep="\t", dtype='unicode')
"""transform list from emsembl2ncbiid.py"""
emsembl2ncbiid = pd.read_csv("ensembl2ncbi_concat.csv", sep=",")
fin_profile = emsembl2ncbiid.join(gene_profile, on="Ensembl_ID").drop(columns=['HGNC symbol', 'Nomenclature_status', 'tax_id', 'type_of_gene',]).set_index("GeneID")
rawdata = fin_profile.drop(columns=["Ensembl_ID"]).set_index("Symbol").T

#rawdata = (pd.read_table("merge table.csv")).T
rawdata.shape
data = rawdata.values[:, 1:]
data.shape

# Here are labels for cancer types. its 
labels = [i for i in rawdata][1:]
print(len(labels))
np.save("labels.npy", labels)
print("examples:", labels[:10])

# =============================================================================
# Making train/valid/test splits here
# =============================================================================
#df = pd.read_table("../data/rawdata/TCGA/TCGA_cancer_types.tsv")

# Basically making a one hot encoded truth matrix. so you would expect a 9092 by 33 matrix. 
#temp = df["Cancertype"].values
temp = (["WT"]*120+["KIRP"]*288)
lbs = list(set(temp))
lbmap = dict([(lbs[i], i) for i in range(len(lbs))])

y = np.zeros((len(temp), len(lbs)))
for i in range(len(temp)):
    y[i, lbmap[temp[i]]] = 1

# Double checking with an assert statement.
assert y.shape[0] ==  np.sum(y)

# Generating indexes to split on. We are using 640 test, 1280 validation, and about 7000 training samples.
indexes = np.arange(y.shape[0])
np.random.shuffle(indexes)
train = indexes[:-272]
valid = indexes[:]
test = indexes[-136:]
len(train), len(valid), len(test)


np.save("trainX.npy", data[train])
np.save("validX.npy", data[valid])
np.save("testX.npy", data[test])

np.save("trainY.npy", y[train])
np.save("validY.npy", y[valid])
np.save("testY.npy", y[test])


# =============================================================================
# Making adjacency matrix
# =============================================================================
rawppi = open("BIOGRID-ORGANISM-Homo_sapiens-3.5.170.mitab.txt")
ppi = rawppi.readlines()
# intx is a class that stores ppi
temp = utils.intx(ppi[123])
print(temp)

# Double checking if it works
print("SSR3" in temp)
print(temp.isInteracting("PRRC2A", "SSR3"))
print(temp.isInteracting("SSR3", "PRRC2A"))
print(temp.isInteracting("SSR3", "dmy"))

# Loading in all interactions
interactions = [utils.intx(i) for i in ppi[1:]]
print("# of interactions:", len(interactions))

# How many iteraction does SERF2 have?
sum(["SERF2" in i for i in interactions])

# Basically checking if a label is involved in an interaction (0th)
print(interactions[0])
inds = [i for i in range(len(labels)) if labels[i] in interactions[0]]
print(inds)

# We are generating a 20000 by 20000 ppi matrix. This may take a while but you need to only do it once.
ppi_matrix = np.zeros((len(labels), len(labels)))

discarded = []
# For each interaction,
for interaction in interactions:
    # Check which genes participate in interactions
    # There are some weird cases where more than 2 genes participate in a single interactions 
    inds = [i for i in range(len(labels)) if labels[i] in interaction]
    temp = []
    
    # Get a tuple of indecies to fill. Do not fill in self-interaction.
    for i in inds:
        for j in inds:
            if i!=j:
                temp.append([i, j])
                temp.append([j, i])
                
    # Fill the matrix.
    for t in temp:
        ppi_matrix[t[0], t[1]] = 1
    
    # Just keep track of what kinds of interactions are being discarded.
    if len(temp) == 0:
        discarded.append(interaction)
                
    # Print interactions that are weird.
    if len(inds)<2:
        #print [labels[j] for j in inds], interaction, temp
        pass

np.save("ppi2.npy", ppi_matrix)

ppi_matrix = np.load("ppi2.npy")
print("Shape:", ppi_matrix.shape)
print("Sparcity:", np.sum(ppi_matrix)/(ppi_matrix.shape[0]**2))
