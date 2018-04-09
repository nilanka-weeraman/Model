# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 23:11:56 2018

@author: nilanka_03234
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
import cx_Oracle
import pandas as pd
import os

os.chdir("D:\\D\\BI\\2018-04\\")
# no of attributes
na=int(19)
"""
\\\\ TODO define na  define Tx define m
"""
def fetchData(query):
    con = cx_Oracle.connect('olap_edw/olap#rep#admin@172.26.6.221/mis')
    df=pd.read_sql(query,con)
    return df

def rearrange(dfx):
    """
    rearrange dataframe to a compatible shape with LSTM
    shape should be ( no_of_samples, no_of_dims, no_of_stepsinLSTM )
    """
    df_x=dfx
    dfxm=df_x[['WK1', 'WK2', 'WK3', 'WK4', 'WK5', 'WK6',
    'WK7', 'WK8', 'WK9', 'WK10', 'WK11', 'WK12', 'WK13', 'WK14', 'WK15',
    'WK16', 'WK17', 'WK18', 'WK19', 'WK20']].as_matrix()

    dfxmr=dfxm.reshape(int(np.shape(df_x)[0]/na),na,20)
    return dfxmr

def setupModel():
    """
    Build the model graph
    Reference https://keras.io/getting-started/sequential-model-guide/ & DeepLearning course notes of Andrew NG
    """
    #model = Sequential()
    #model.add(LSTM(128,input_shape=(None,20),return_sequences=False))
    #model.add(Dropout(0.5))
    #model.add(Dense(1, activation='sigmoid'))
    #model.compile(loss='binary_crossentropy',
    #              optimizer='adam',
    #              metrics=['accuracy'])
    #model.summary()
    
    model = Sequential()
    model.add(LSTM(64,input_shape=(None,20),return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64,return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
              
    return model


def trainModel(model,dfxmr,dfy):
    """
    1. assign training features and target from fetch & reshaped data
    2. train the model
    """
    x_train = dfxmr
    y_train = dfy['TARGET'].as_matrix()
    model.fit(x_train, y_train, batch_size=16, epochs=10)

def scoreModel(model,x_test,y_test):
    score = model.evaluate(x_test, y_test, batch_size=16)


"""
MAIN EXECUTION STEPS - TRAINING 
"""
query_x= """select * from sas_user.dtv_chdl_tx_array_fin_train1 
         order by 2,1"""
query_y= """SELECT a.contract_id,target
    FROM (SELECT DISTINCT contract_id 
    FROM sas_user.dtv_chdl_tx_array_fin_train1
    ORDER BY 1) a,
    sas_user.dtv_chdl_base b
    WHERE a.contract_id = b.contract_id
    ORDER BY 1"""


dfx=fetchData(query_x)
dfy=fetchData(query_y)

dfxmr=rearrange(dfx)

model=setupModel()
#train the model
trainModel(model=model,dfxmr=dfxmr,dfy=dfy)
#predict the outcome for training sample
model.predict(x=x_train)

pd.DataFrame( model.predict(x=x_train)).to_csv('x_train_result3.csv')
dfy.to_csv('dfy3.csv')

scoreModel(model=model,x_test=x_train,y_test=y_train)


""" Testing """

query_x_test= """select * from sas_user.dtv_chdl_tx_array_fin_test2 
         order by 2,1"""
query_y_test= """SELECT a.contract_id,target
    FROM (SELECT DISTINCT contract_id 
    FROM sas_user.dtv_chdl_tx_array_fin_test2
    ORDER BY 1) a,
    sas_user.dtv_chdl_base b
    WHERE a.contract_id = b.contract_id
    ORDER BY 1"""

dfx_test=fetchData(query_x_test)
dfy_test=fetchData(query_y_test)

dfxmr_test=rearrange(dfx_test)

np.shape(dfxmr_test)
np.shape(x_train)
np.shape(dfy_test)

pd.DataFrame( model.predict(x=dfxmr_test)).to_csv('x_test_batch2_result.csv')
dfy_test.to_csv('y_test_batch2_actual.csv')

#####################

np.shape(pd.DataFrame( model.predict(x=dfxmr_test))

pd.concat([dfx_test[dfx_test['TX_TYPE']=='discon'].reset_index(drop=True),
          dfy_test.reset_index(drop=True),
          pd.DataFrame(model.predict(x=dfxmr_test)).reset_index(drop=True)],axis=1).to_csv('random_check_batch2_discon.csv')

dfx_test.groupby(dfx_test['TX_TYPE']).count()['WK1']

########################################################################################
########################################################################################
########################################################################################

dfx_test[dfx_test['TX_TYPE']=='bills'].head(5)
dfy_test.head(5)
test_all=pd.concat([dfx_test[dfx_test['TX_TYPE']=='bills'].reset_index(drop=True),
           dfy_test.reset_index(drop=True)],axis=1)

test_all.head(10)

dfx_test.


def setupModel():
    """
    Build the model graph
    Reference https://keras.io/getting-started/sequential-model-guide/ & DeepLearning course notes of Andrew NG
    """
    model = Sequential()
    model.add(LSTM(128,input_shape=(None,20),return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    
    return model