# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 11:48:52 2018

@author: nilanka_03234
"""

import numpy as np
import pandas as pd
np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
import os
import cx_Oracle
import pickle

con = cx_Oracle.connect('olap_edw/olap#rep#admin@172.26.6.221/mis')

#read data from Oracle DWH. This is supposed to be faster ;-)
def read_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute( query )
        names = [ x[0] for x in cursor.description]
        rows = cursor.fetchall()
        return pd.DataFrame( rows, columns=names)
    finally:
        if cursor is not None:
            cursor.close()
            
def DTV_Churn(input_shape):
    """
    Function creating the DTV Churn model's graph.
    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    Returns:
    model -- a model instance in Keras
    """
           
    ip = Input(shape=np.shape(input_shape),dtype='int32')
    X = LSTM(12)(ip)
    #X = LSTM(12)(X)   
    #output is a desn
    X = Dense(1)(X)
    # Add a tanh activation, since it has a steep gradient and useful during training
    X = Activation('tanh')(X)
    
    model = Model(inputs = ip, outputs = X)
        
    return model

os.chdir("D:\\D\\BI\\2018-04\\")
maxLen = 20

dataset  = read_query(con,'select * from sas_user.dtv_chdl_tx_array_fin_fil1_1');
dataset.head(5)
dataset.to_csv('dataset')
ds_mat=dataset.as_matrix()
ds_mat.shape

type(dataset)


model = DTV_Churn(([None,20,19]))
