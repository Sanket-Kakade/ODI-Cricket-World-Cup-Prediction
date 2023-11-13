# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 23:37:37 2023

@author: sanket
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score,accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Activation
from tensorflow.keras.initializers import HeNormal, RandomNormal, GlorotNormal 
from tensorflow.keras.optimizers import Adadelta, Adam, RMSprop
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from sklearn.metrics import roc_auc_score
import sys
tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()

#%%
all_odi_data_loc= "K:\\Sanket-datascience\\CWC_prediction\\Data\\"
all_odi_data_file= "feat_df_more_mts.pkl"
feat_data= pd.read_pickle(all_odi_data_loc+all_odi_data_file)
feat_data2= feat_data[feat_data['date']>'2015/1/1']
feat_data2.drop('date',axis=1,inplace=True)
#%%

feat_data2.fillna(0,inplace=True)

#%%
label_encoder = LabelEncoder()
feat_data2['winner_le'] = np.where(feat_data2['winner_encoded']=='team_1',0,1)
feat_data2.drop('winner_encoded',axis=1,inplace=True)
dummy_cols= ['team_1_Afghanistan', 'team_1_Australia',
'team_1_Bangladesh', 'team_1_Bermuda', 'team_1_Canada',
'team_1_England', 'team_1_Hong Kong', 'team_1_India', 'team_1_Ireland',
'team_1_Jersey', 'team_1_Kenya', 'team_1_Namibia', 'team_1_Nepal',
'team_1_Netherlands', 'team_1_New Zealand', 'team_1_Oman',
'team_1_Pakistan', 'team_1_Papua New Guinea', 'team_1_Scotland',
'team_1_South Africa', 'team_1_Sri Lanka',
'team_1_United Arab Emirates', 'team_1_United States of America',
'team_1_West Indies', 'team_2_Australia',
'team_2_Bangladesh', 'team_2_Bermuda', 'team_2_Canada',
'team_2_England', 'team_2_Hong Kong',
'team_2_India', 'team_2_Ireland', 'team_2_Kenya', 'team_2_Namibia',
'team_2_Nepal', 'team_2_Netherlands', 'team_2_New Zealand',
'team_2_Oman', 'team_2_Pakistan', 'team_2_Papua New Guinea',
'team_2_Scotland', 'team_2_South Africa', 'team_2_Sri Lanka',
'team_2_United Arab Emirates', 'team_2_United States of America',
'team_2_West Indies', 'team_2_Zimbabwe']
feat_data2.drop(dummy_cols, axis=1,inplace=True,errors='ignore')
#%%
x_train,x_test, y_train, y_test= train_test_split(feat_data2.drop('winner_le',axis=1),feat_data2['winner_le'],test_size=0.2)
# train, test= feat_data2[feat_data2['date']<='2020/1/1'], feat_data2[feat_data2['date']>'2020/1/1']
# train.drop('date',axis=1,inplace=True)
# test.drop('date',axis=1,inplace=True)
# x_train,y_train=  train.drop('winner_le',axis=1), train[['winner_le']]
# x_test, y_test= test.drop('winner_le',axis=1), test[['winner_le']]

#%%
lr= LogisticRegression(max_iter=10000, penalty='l1', solver='liblinear',random_state=5)
lr.fit(x_train, y_train)
y_pred= lr.predict(x_test)
y_train_pred= lr.predict(x_train)
print ("LR")
print(f1_score(y_train, y_train_pred))
print(accuracy_score(y_train, y_train_pred))

print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
#%%

#%%
coef_l1= lr.coef_.ravel()
feats= x_train.columns.values.ravel()

coef_df= pd.DataFrame({'feat':feats, 'coef':coef_l1})
#%%
rf= RandomForestClassifier()
rf.fit(x_train, y_train)
y_pred= rf.predict(x_test)
y_train_pred= rf.predict(x_train)
print ("RF")
print(f1_score(y_train, y_train_pred))
print(accuracy_score(y_train, y_train_pred))

print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

#%%
gbc= GradientBoostingClassifier(learning_rate=1e-3, n_estimators=500,max_depth=4)
gbc.fit(x_train, y_train)
y_pred= gbc.predict(x_test)
y_train_pred= gbc.predict(x_train)
print ("XGB")
print(f1_score(y_train, y_train_pred))
print(accuracy_score(y_train, y_train_pred))
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
#%%
svc= SVC(kernel='rbf',probability=True)
svc.fit(x_train, y_train)
y_pred= svc.predict(x_test)
y_train_pred= svc.predict(x_train)
print('svc')
print(f1_score(y_train, y_train_pred))
print(accuracy_score(y_train, y_train_pred))
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

#%%
initializer1 = HeNormal()
initializer2 = RandomNormal()
initializer3 = GlorotNormal()
ini_ls= [initializer1,initializer2,initializer3]
lr_ls= [1e-3,1e-4]
ep_ls= [200,500,1000]
l1_neuron_ls= [30,50]
l2_neuron_ls= [15,30]

tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()
optimizer = tf.keras.optimizers.Adam(0.00001)

# define the keras model
nn_model = Sequential()

nn_model.add(Dense(30, input_shape=(x_train.shape[1],), activation='relu',kernel_initializer= initializer1))
nn_model.add(Dropout(0.1,seed=5))

nn_model.add(Dense(25, activation='relu',kernel_initializer= initializer1))
nn_model.add(Dense(1, activation='sigmoid'))
nn_model.compile(loss='poisson', optimizer=optimizer, metrics=['accuracy'])
history= nn_model.fit(x_train, y_train, epochs=1000, batch_size=8,verbose=0,validation_data= (x_test,y_test))
_, accuracy = nn_model.evaluate(x_train, y_train,)
y_pred_proba = nn_model.predict(x_test)
y_train_pred_proba = nn_model.predict(x_train)
# round predictions 
y_pred = [round(x[0]) for x in y_pred_proba]
y_train_pred = [round(x[0]) for x in y_train_pred_proba]

print ("NN")

print(f1_score(y_train, y_train_pred))
print(accuracy_score(y_train, y_train_pred))
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
#%%
initializer1 = HeNormal()
initializer2 = RandomNormal()
initializer3 = GlorotNormal()

space = {
     'choice': hp.choice('num_layers',
                     [{'layers':'two', },
                     {'layers':'three',
                     'units3': scope.int(hp.quniform('units3', 4,128,8)), 
                     'dropout3': hp.quniform('dropout3', .10,.50,0.05)}
                     ]),

            'units1': scope.int(hp.quniform('units1', 16,128,8)),
            'units2': scope.int(hp.quniform('units2', 8,128,8)),

            'dropout1': hp.quniform('dropout1', .10,.50,0.05),
            'dropout2': hp.quniform('dropout2',  .10,.50,0.05),

            # 'batch_size' : hp.uniform('batch_size', 4,64),
            'batch_size': scope.int(hp.quniform('batch_size', 4,48,4)),

            'nb_epochs' :  scope.int(hp.quniform('nb_epoch', 100,1000,50)),
            'learning_rate':hp.choice('learning_rate',[1e-4,1e-5,1e-6]),
            'activation': 'relu'
        }

def f_nn(params):   

    print ('Params testing: ', params)
    model = Sequential()
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
    # nn_model.add(Dense(30, input_shape=(x_train.shape[1],), activation='relu',kernel_initializer= initializer1))
    optimizer = tf.keras.optimizers.Adam(params['learning_rate'])
    model.add(Dense(params['units1'], input_dim = x_train.shape[1],activation=params['activation'] )) 
    model.add(Dropout(params['dropout1']))

    model.add(Dense(params['units2'], kernel_initializer = "he_normal", activation=params['activation'])) 
    model.add(Dropout(params['dropout2']))

    if params['choice']['layers']== 'three':
        model.add(Dense(params['choice']['units3'], kernel_initializer = "he_normal",activation=params['activation'])) 
        model.add(Dropout(params['choice']['dropout3']))    

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='poisson', optimizer=optimizer, metrics= 'accuracy')

    model.fit(x_train,y_train,epochs=params['nb_epochs'],batch_size=params['batch_size'],callbacks=[callback],validation_data=(x_test,y_test),verbose = 1)

    pred_auc =model.predict(x_test, batch_size = 128, verbose = 0)
    acc= roc_auc_score(y_test, pred_auc)
    
    print('AUC:', acc)
    sys.stdout.flush() 
    return {'loss': -acc, 'status': STATUS_OK}
trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=150, trials=trials)
print('best: ', best)


#%%
from hyperopt import space_eval
best_param_nn= space_eval(space, best)

# best_param_nn= {'activation': 'relu',
#  'batch_size': 8,
#  'choice': {'layers': 'two'},
#  'dropout1': 0.1,
#  'dropout2': 0.4,
#  'learning_rate': 0.1e-5,
#  'nb_epochs': 1000,
#  'units1': 56,
#  'units2': 40}

#%%

optimizer = tf.keras.optimizers.Adam(best_param_nn['learning_rate'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

ho_model = Sequential()
# nn_model.add(Dense(30, input_shape=(x_train.shape[1],), activation='relu',kernel_initializer= initializer1))

ho_model.add(Dense(best_param_nn['units1'], input_dim = x_train.shape[1],kernel_initializer = "he_normal") ) 
ho_model.add(Activation(best_param_nn['activation']))
ho_model.add(Dropout(best_param_nn['dropout1']))

ho_model.add(Dense(best_param_nn['units2'], kernel_initializer = "he_normal")) 
ho_model.add(Activation(best_param_nn['activation']))
ho_model.add(Dropout(best_param_nn['dropout2']))

if best_param_nn['choice']['layers']== 'three':
    ho_model.add(Dense(best_param_nn['choice']['units3'], kernel_initializer = "he_normal")) 
    ho_model.add(Activation(best_param_nn['activation']))
    ho_model.add(Dropout(best_param_nn['choice']['dropout3']))    

ho_model.add(Dense(1))
ho_model.add(Activation('sigmoid'))
ho_model.compile(loss='binary_crossentropy', optimizer=optimizer)

history= ho_model.fit(x_train,y_train,epochs=1000,batch_size=best_param_nn['batch_size'],callbacks=None,verbose =0,validation_data=(x_test,y_test))

y_pred_proba = ho_model.predict(x_test)
y_train_pred_proba = ho_model.predict(x_train)
# round predictions 
y_pred = [round(x[0]) for x in y_pred_proba]
y_train_pred = [round(x[0]) for x in y_train_pred_proba]


print ("NN Hyper")

print(f1_score(y_train, y_train_pred))
print(accuracy_score(y_train, y_train_pred))
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
#%%
import xgboost as xgb

dtrain = xgb.DMatrix(x_train, label=y_train)
dvalid = xgb.DMatrix(x_test, label=y_test)

space_xgb = {
        'num_boost_round': scope.int(hp.quniform('num_boost_round', 100, 1000,25)),
        'eta': hp.quniform('eta', 1e-5, 0.1, 0.002),
        'max_depth':  scope.int(hp.quniform('max_depth', 1,7,1)),
        'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 6,1)),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'eval_metric': 'auc',
        'objective': hp.choice('objective',['binary:logistic','binary:hinge']),
        'nthread': 4,
        'booster': 'gbtree',
        'tree_method': 'exact',
        'silent': 1,
        'seed': 5
        }
def f_xgb(params):

    gbm_model = xgb.train(params, dtrain,num_boost_round= params['num_boost_round'],verbose_eval=False)
    predictions = gbm_model.predict(dvalid,ntree_limit=gbm_model.best_iteration + 1)
    score = roc_auc_score(y_test, predictions)
    loss = 1 - score
    print('AUC:', score)

    return {'loss': loss, 'status': STATUS_OK}

trials_xgb = Trials()
best_xgb = fmin(f_xgb, space_xgb, algo=tpe.suggest, max_evals=800,trials=trials_xgb)

best_xgb_param_dic= space_eval(space_xgb, best_xgb)
#%%
best_xgb_param_dic= {
    'booster': 'gbtree',
 'colsample_bytree': 0.75,
 'eta': 0.00002,
 'eval_metric': 'auc',
 'gamma': 0.0,
 'max_depth': 4,
 'min_child_weight': 2,
 'nthread': 4,
 'num_boost_round': 1000,
 'objective': 'binary:logistic',
 'seed': 5,
 'silent': 1,
 'subsample': 0.8,
 'tree_method': 'exact'}
#%%
gbm_model = xgb.train(best_xgb_param_dic, dtrain,verbose_eval=True)
y_pred_proba_xgb= gbm_model.predict(dvalid)#ntree_limit=gbm_model.best_iteration + 1)
y_train_pred_proba = gbm_model.predict(dtrain)#,ntree_limit=gbm_model.best_iteration + 1)
# round predictions 
y_pred = np.round(y_pred_proba_xgb)
y_train_pred = np.round(y_train_pred_proba)


print ("XGB Hyper")

print(f1_score(y_train, y_train_pred))
print(accuracy_score(y_train, y_train_pred))
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print (roc_auc_score(y_test, y_pred_proba_xgb))

#%%
from sklearn.ensemble import VotingClassifier
model1 = lr
model2= svc
model3= rf
model = VotingClassifier(estimators=[('lr', model1),('svc',model2)], voting='soft')
model.fit(x_train,y_train)


y_pred_ens= model.predict_proba(x_test,)
y_pred_nn= ho_model.predict(x_test,)
y_pred_xgb= gbm_model.predict(dvalid)

y_train_pred_ens= model.predict_proba(x_train)
y_train_pred_nn= ho_model.predict(x_train)

y_train_pred_xgb= gbm_model.predict(dtrain)

y_train_pred_comb= np.array([y_train_pred_ens[:,1],y_train_pred_nn[:,0],y_train_pred_xgb[:]]).T
y_train_pred_comb= np.array([y_train_pred_ens[:,1],y_train_pred_xgb[:]]).T

y_train_pred_comb= np.average(y_train_pred_comb,axis=1)

y_pred_comb= np.array([y_pred_ens[:,1],y_pred_nn[:,0],y_pred_xgb[:]]).T
y_pred_comb= np.array([y_pred_ens[:,1],y_pred_xgb[:]]).T
y_pred_comb= np.average(y_pred_comb,axis=1)
y_pred_comb = np.round(y_pred_comb)
y_train_pred_comb = np.round(y_train_pred_comb)


print ("Ens")

print(f1_score(y_train, y_train_pred_comb))
print(accuracy_score(y_train, y_train_pred_comb))
print(f1_score(y_test, y_pred_comb))
print(accuracy_score(y_test, y_pred_comb))
#%%
# Saving models
import pickle   
pickle_out1 = open("K:\Sanket-datascience\CWC_prediction\Models\lr_model.pkl", "wb")    
pickle.dump(lr, pickle_out1)    
pickle_out1.close()
pickle_out2 = open("K:\Sanket-datascience\CWC_prediction\Models\svc_model.pkl", "wb")    
pickle.dump(svc, pickle_out2)    
pickle_out2.close()
pickle_out3 = open("K:\Sanket-datascience\CWC_prediction\Models\gbm_model.pkl", "wb")    
pickle.dump(gbm_model, pickle_out3)    
pickle_out3.close()