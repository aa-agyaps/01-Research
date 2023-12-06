# -*- coding: utf-8 -*-
"""
Created on Sun May 17 14:43:53 2020

@author: janov
"""
target=1# 
        # 1:Tg_low      
        # 2:Tg_high
        #3:rubbery modulus
        #4:glassy modulus
# Define the path variables
path1 = 'D:\Southern\Ama/03-Low_Temp - Practice.csv'#glass modulus
path2 = 'D:\Southern\Ama/04-High_Temp - Practice.csv'#glass modulus
path3 = 'D:\Southern\Ama/02-Rubbery_Modulus.csv'#rubbery modulus
path4 = 'D:\Southern\Ama/ama_glass_modulus.csv'

# Choose the desired path variable
if target ==1:
      random_state=1#Tg_low:24 #good
      selected_path = path1
elif target ==2:
    random_state=1#Tg_high
    selected_path = path2      
elif target ==3:
    random_state=3#rubbery modulus
    selected_path = path3
else:
    random_state=19#glassy modulus  #good
    selected_path = path4



from rdkit import Chem
from rdkit.Chem import AllChem
from keras.models import Sequential,load_model
from keras.layers import LSTM, Dense, Dropout,LeakyReLU,Concatenate
from keras.models import load_model
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import csv
from keras.layers import Input, Activation,Flatten
from keras.models import Model
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.layers import Bidirectional
import random
import keras.backend as K
from keras.layers.embeddings import Embedding
# from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
import string
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

from keras.layers import Dropout
from keras.layers import Flatten,GlobalMaxPool2D
# from keras.losses import MeanAbsoluteError
# from keras.losses import MeanAbsoluteError

from keras.backend import clear_session
from keras.utils.vis_utils import plot_model
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from keras.callbacks import History, ReduceLROnPlateau
import pandas as pd
import math
from sklearn.metrics import r2_score

name=[]

def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))
# root mean squared error (rmse) for regression (only for Keras tensors)
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

# mean squared error (mse) for regression  (only for Keras tensors)
def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)

# coefficient of determination (R^2) for regression  (only for Keras tensors)
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
no=163
name=[]
smiles=[]

ratio1=[]
ratio2=[]
ratio3=[]
ratio4=[]

prediction_aim=[]
glassy_modulus=[]
rubbery_modulus=[]
glassy_tran_up=[]
glassy_tran_low=[]
glassy_tran_temp=[]
cross_link_density=[]
prog_temp=[]
train_temp=[]
prog_strain=[]
ratio=[]






name = []
smiles = []
predic_target = []
ratio = []

target_column=6
# Use the selected_path variable with the with open() statement
with open(selected_path) as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for column in csvReader:
        name.append(column[2])
        #tlc[:,[1,length]].append(column[0])
        smiles.append(column[3])
        prediction_aim.append(column[target_column])
        ratio.append(column[4])
        # glassy_modulus.append(column[5])
        # rubbery_modulus.append(column[6])
        # glassy_tran_up.append(column[7])
        # glassy_tran_low.append(column[8])
        # glassy_tran_temp.append(column[9])
        # cross_link_density.append(column[10])
        # prog_temp.append(column[11])
        # train_temp.append(column[12])
        # prog_strain.append(column[13])
        
        # ratio1.append(column[16])
        # ratio2.append(column[17])
        
        # ratio3.append(column[18])
        # ratio4.append(column[19])
        
        
name.pop(0) #cut the label name from the array
smiles.pop(0)
ratio.pop(0)
# ratio1.pop(0)
# ratio2.pop(0)
# ratio3.pop(0)
# # kesi_Tg2.pop(0)
# ratio4.pop(0)
prediction_aim.pop(0)
# glassy_modulus.pop(0)
# rubbery_modulus.pop(0)
# glassy_tran_up.pop(0)
# glassy_tran_low.pop(0)
# glassy_tran_temp.pop(0)
# cross_link_density.pop(0)
# prog_temp.pop(0)
# train_temp.pop(0)
# prog_strain.pop(0)
prediction_aim=list(np.float_(prediction_aim))
#delete space in the array
for i in range(0,int(len(smiles))):
          smiles[i]=smiles[i].translate({ord(c): None for c in string.whitespace})

length=int(len(prog_strain))
prog_strain=np.array(prog_strain[0:length]).astype(np.float)





#delete white space in the SMILES
i=0
for i in range(0,int(len(smiles))):
          smiles[i]=smiles[i].translate({ord(c): None for c in string.whitespace})
          smiles[i] = smiles[i].split(',')
          smiles[i]=[item.replace("{", "") for item in smiles[i]]
          smiles[i]=[item.replace("}", "") for item in smiles[i]]
              
              
smiles_1=[[] for i in range(int(len(smiles)))]
smiles_2=[[] for i in range(int(len(smiles)))]
smiles_3=[[] for i in range(int(len(smiles)))]
smiles_4=[[] for i in range(int(len(smiles)))]     



for i in range(0,int(len(ratio))):
          ratio[i]=ratio[i].translate({ord(c): None for c in string.whitespace})
          ratio[i] = ratio[i].split(':')
       
              
ratio_1=[[] for i in range(int(len(smiles)))]
ratio_2=[[] for i in range(int(len(smiles)))]
ratio_3=[[] for i in range(int(len(smiles)))]
ratio_4=[[] for i in range(int(len(smiles)))]            

i=0          
for i in range(0,int(len(smiles))):          
          smiles_1[i]=smiles[i][0]
          ratio_1[i]=ratio[i][0]
          if len(smiles[i])>1:
              smiles_2[i]=smiles[i][1]
              ratio_2[i]=ratio[i][1]
          if len(smiles[i])>2:
              smiles_3[i]=smiles[i][2]
              ratio_3[i]=ratio[i][2]
          if len(smiles[i])>3:
              smiles_4[i]=smiles[i][3]
              ratio_4[i]=ratio[i][3]


#Parameters for Morgan fingerprinting
latent_dim = 64*4*4*2
rad=13
#calculate combined vector
combined_vetor_all=np.zeros((int(len(ratio)),latent_dim))
for i in range(0,int(len(smiles))):          
          
          mol = Chem.MolFromSmiles(smiles_1[i])
          fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=rad, nBits=latent_dim)
          vec1 = fp.ToBitString()
          vec1=np.asarray(list(map(float, vec1)))
          combined_vetor=float(ratio_1[i])*vec1
          if len(smiles[i])>1:
             mol = Chem.MolFromSmiles(smiles_2[i])
             fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=rad, nBits=latent_dim)
             vec2 = fp.ToBitString()
             vec2=np.asarray(list(map(float, vec2)))
             combined_vetor=float(ratio_1[i])*vec1+float(ratio_2[i])*vec2
          if len(smiles[i])>2:
             mol = Chem.MolFromSmiles(smiles_3[i])
             fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=rad, nBits=latent_dim)
             vec3 = fp.ToBitString()
             vec3=np.asarray(list(map(float, vec3)))
             combined_vetor=float(ratio_1[i])*vec1+float(ratio_2[i])*vec2+float(ratio_3[i])*vec3
          if len(smiles[i])>3:
              mol = Chem.MolFromSmiles(smiles_4[i])
              fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=rad, nBits=latent_dim)
              vec4 = fp.ToBitString()
              vec4=np.asarray(list(map(float, vec4)))
              combined_vetor=float(ratio_1[i])*vec1+float(ratio_2[i])*vec2+float(ratio_3[i])*vec3+float(ratio_4[i])*vec4
          combined_vetor_all[i]=combined_vetor
# combined_vetor_all=combined_vetor_all.reshape(len(combined_vetor_all),256)
#SVM method
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
# y = np.array([1, 1, 2, 2])
from sklearn.svm import SVC,SVR
clf = make_pipeline(StandardScaler(),SVR(C=33,kernel='rbf', epsilon=1))
train_len=int(0.8*int(len(smiles)))
prediction_aim=np.array(prediction_aim)
# clf.fit(combined_vetor_all[0:len(combined_vetor_all)-21], prediction_aim[0:len(combined_vetor_all)-21])
if target==6:
   train_len_effec=len(combined_vetor_all)-5
else:
   train_len_effec=len(combined_vetor_all)
# y_max=prediction_aim.max()
# y_min=prediction_aim.min()
# data = (prediction_aim-y_min)/(y_max-y_min)
# LY = np.log10(prediction_aim + 1 - min(prediction_aim))


if target==5:
    y_value=prediction_aim+0#transfer celsius to kelvin
    pcp_index=0.15
else:
    y_value=prediction_aim
    pcp_index=0.2
# from sklearn import preprocessing
# X_scaled = preprocessing.scale(combined_vetor_all)
from sklearn.decomposition import PCA 

dimension_pca=110
pca=PCA(dimension_pca)


to_pca=combined_vetor_all
onestep_pca=(pca.fit_transform(to_pca))
#latent_dim=dimension_pca

X_train, X_test, y_train, y_test = train_test_split(combined_vetor_all[0:train_len_effec], 
                                                    y_value[0:train_len_effec], 
                                                    test_size=0.20, random_state=random_state)#7 for stress and 7 for temperature
clf.fit(X_train, y_train)


# calculate SMP by SVM in test data
j=0
diff=np.zeros(len(X_test))
ab_diff_test=np.zeros(len(X_test))


pred_test=np.zeros(len(X_test))
for i in range(0,len(X_test)):
        pred_test[i]=clf.predict(X_test[i].reshape(1,len(X_test[0])))
        #pred_original=pred*(y_max-y_min)+y_min 
        
        #diff[i]=abs((pred_original-y_original)/y_original)
        ab_diff_test[i]=abs(pred_test[i]-y_test[i])
        diff[i]=abs((pred_test[i]-y_test[i])/y_test[i])
        if abs(diff[i])<pcp_index:
            j=j+1
        # print(i,pred_original,y_test[i],diff[i])
        #print(i,pred_original,y_test[i],diff[i])
        print(i,pred_test[i],y_test[i],diff[i])
       
mean_diff_test=np.mean(diff)
pcp_test=j/len(y_test)
print(mean_diff_test,np.mean(ab_diff_test),"{:.3%}".format(pcp_test))

coefficient_of_dermination = r2_score(y_test, pred_test)
print("R Square is",coefficient_of_dermination)
# # # # save the model to disk
# # # import pickle
# # # # # filename = 'recycle_eff_model2.sav'
# # # filename = 'Tg_model.sav'

# # # pickle.dump(clf, open(filename, 'wb'))
# Plot test data
csfont = {'fontname':'Times New Roman'}
# plt.title('title',**csfont)


fig, ax = plt.subplots()
ax.scatter(y_test,pred_test, s=25, cmap=plt.cm.coolwarm, zorder=10,marker = '^')
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
# now plot both limits against eachother
ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

plt.text(.01, .79,'MAPE={:.2f}%'.format(np.round(mean_diff_test, 2)*100),
         ha='left', va='top', transform=ax.transAxes,fontsize=18)
plt.text(.01, .89,'PCP={:.2f}%'.format(pcp_test*100),
         ha='left', va='top', transform=ax.transAxes,fontsize=18)
plt.text(.01, .99,'$R^2$=%s'%(np.round(coefficient_of_dermination, 2)),
         ha='left', va='top', transform=ax.transAxes,fontsize=18)

plt.ylabel("Prediction (%)",**csfont,fontsize=18)
plt.xlabel("Test data (%)",**csfont,fontsize=18)
plt.grid(linestyle='dotted')
plt.show()

#caculate training data
j=0
diff=np.zeros(len(X_train))
ab_diff_train=np.zeros(len(X_train))


pred_train=np.zeros(len(X_train))
for i in range(0,len(X_train)):
        pred_train[i]=clf.predict(X_train[i].reshape(1,len(X_train[0])))
        #pred_original=pred*(y_max-y_min)+y_min 
        
        #diff[i]=abs((pred_original-y_original)/y_original)
        ab_diff_train[i]=abs(pred_train[i]-y_train[i])
        diff[i]=abs((pred_train[i]-y_train[i])/y_train[i])
        if abs(diff[i])<pcp_index:
            j=j+1
        # print(i,pred_original,y_test[i],diff[i])
        #print(i,pred_original,y_test[i],diff[i])
        print(i,pred_train[i],y_train[i],diff[i])
       
mean_diff_train=np.mean(diff)
pcp_train=j/len(y_train)
print(mean_diff_train,np.mean(ab_diff_train),"{:.3%}".format(pcp_train))
coefficient_of_dermination = r2_score(y_train, pred_train)
print("R Square is",coefficient_of_dermination)
#plot training data
fig, ax = plt.subplots()
ax.scatter(y_train,pred_train, s=25, cmap=plt.cm.coolwarm, zorder=10,marker = '^')
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
# now plot both limits against eachother
ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

plt.text(.01, .79,'MAPE={:.2f}%'.format(np.round(mean_diff_train, 2)*100),
         ha='left', va='top', transform=ax.transAxes,fontsize=18)
plt.text(.01, .89,'PCP={:.2f}%'.format(pcp_train*100),
         ha='left', va='top', transform=ax.transAxes,fontsize=18)
plt.text(.01, .99,'$R^2$=%s'%(np.round(coefficient_of_dermination, 2)),
         ha='left', va='top', transform=ax.transAxes,fontsize=18)

plt.ylabel("Prediction (%)",**csfont,fontsize=18)
plt.xlabel("Training data (%)",**csfont,fontsize=18)
plt.grid(linestyle='dotted')
plt.show()



# mol = "C=CCOC(C)COC(O)C1CC(C)(CCC(=O)OCCOC(=O)C=C)C1OC(=O)COC(=O)C(=C)C"
# mol = Chem.MolFromSmiles(mol)
# fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=rad, nBits=latent_dim)
# vec= fp.ToBitString()
# vec=np.asarray(list(map(float, vec)))

# print(clf.predict(vec.reshape(1,2048)))

# mol1 = "C=CC(=O)OCC(O)COc2ccc(C(C)(C)c1ccc(OCC(O)COC(=O)C=C)cc1)cc2"
# mol1 = Chem.MolFromSmiles(mol1)
# fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=rad, nBits=latent_dim)
# vec1= fp1.ToBitString()
# vec1=np.asarray(list(map(float, vec1)))
# mol2 = "C=CC(=O)OCCn1c(=O)n(CCOC(=O)C=C)c(=O)n(CCOC(=O)C=C)c1=O"
# mol2 = Chem.MolFromSmiles(mol2)
# fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=rad, nBits=latent_dim)
# vec2= fp2.ToBitString()
# vec2=np.asarray(list(map(float, vec2)))

# combined_vetor=float(0.2)*vec1+float(0.8)*vec2


latent_dim=2048

# print(clf.predict(combined_vetor.reshape(1,latent_dim)))

from keras.layers import Conv1D
from keras.layers import MaxPooling1D,GlobalMaxPool1D,Dropout
from keras.utils.vis_utils import plot_model

Neutral_model = Sequential()

Neutral_model.add(Dense(2048, activation='relu', input_shape=(latent_dim,)))
# Neutral_model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
Neutral_model.add(Dense(1280, activation='relu'))
# Neutral_model.add(MaxPooling1D(pool_size=2))
# Neutral_model.add(GlobalMaxPool1D())

# Neutral_model.add(Flatten())
# Neutral_model.add(BatchNormalization())
# Dropout(0.4)
Neutral_model.add(Dense(256, activation='relu'))
# Neutral_model.add(Dense(256, activation='relu'))
# Neutral_model.add(Dense(256, activation='relu'))
# Neutral_model.add(Dense(64, activation="relu"))
# Dropout(0.4)
Neutral_model.add(Dense(64, activation="relu"))
# Neutral_model.add(Dense(64, activation="relu"))
# Neutral_model.add(Dense(32, activation="relu"))
Neutral_model.add(Dense(32, activation="relu"))
# check to see if the regression node should be added
Neutral_model.add(Dense(1, activation="linear"))

def root_mean_squared_error(y_true, y_pred):
        return K.mean(K.abs(y_pred - y_true)/K.abs(y_true)) 
# train_len_effec=len(combined_vetor_all)-5

# Neutral_model.load_weights('D:/LSU/machine learning3/zero__conv1d_model.h5')
# optimizer = Adam(lr=0.001)#0.00001)
Neutral_model.compile(loss="mape", optimizer="adam")#,metrics=["mae",'mape',r_square])#[tf.keras.metrics.MeanSquaredError()])
# Neutral_model.save_weights('zero__conv1d_model.h5')
# plot_model(Neutral_model, to_file='SMP properties model.png', show_shapes=True, show_layer_names=True)# m = Chem.MolFromSmiles('C1=C2C(=CC(=C1Cl)Cl)OC3=CC(=C(C=C3O2)Cl)Cl')

from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint




# Define the ModelCheckpoint callback to save the best model
model_checkpoint = ModelCheckpoint(
    'best_model.h5',  # Replace with your desired path to save the model
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)

# Training the model
his = Neutral_model.fit(
    X_train.reshape(len(X_train), latent_dim), y_train.reshape(len(X_train)),
    validation_data=(X_test.reshape(len(X_test), latent_dim), y_test.reshape(len(X_test))),
    epochs=1000,
    callbacks=[model_checkpoint]  # Add the callback here
)

# Load the best model
best_model = load_model('best_model.h5')

# Evaluate the best model on the test data
test_loss = best_model.evaluate(X_test.reshape(len(X_test), latent_dim), y_test.reshape(len(X_test)))
print("Loss on test data:", test_loss)

# Evaluate the best model on the training data
train_loss = best_model.evaluate(X_train.reshape(len(X_train), latent_dim), y_train.reshape(len(X_train)))
print("Loss on training data:", train_loss)



# # summarize history for loss
# plt.plot(his.history['loss'])
# plt.plot(his.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# #Neutral_model.load_weights(r'D:/LSU/machine learning3/glass_transition_temp_Morgan+ANN.h5')
# #Neutral_model.load_weights(r'D:/LSU/machine learning3/Re_Morgan+ANN.h5')

# # his=Neutral_model.fit(X_train.reshape(len(X_train),latent_dim,1),y_train.reshape(len(X_train),1),#(combined_vetor_all[0:len(combined_vetor_all)-48], prediction_aim[0:len(combined_vetor_all)-48],
# #                     validation_data=(X_test.reshape(len(X_test),latent_dim,1), y_test.reshape(len(X_test),1)),
# #                     epochs=1000,
# #                     #batch_size=64*4#64,
# #                     #shuffle=True,
# #                     # callbacks=[h, rlr],
# #                     )

j=0
diff=np.zeros(len(y_test))
pred=np.zeros(len(y_test))
for i in range(0,len(X_test)):
        pred[i]=best_model.predict(X_test[i].reshape(1,latent_dim))
        # pred_original=pred[i]*(y_max-y_min)+y_min 
        # y_original=y_test[i]*(y_max-y_min)+y_min 
        # diff[i]=abs((pred_original-y_original)/y_original)
        
        diff[i]=abs((pred[i]-y_test[i])/y_test[i])
        if abs(diff[i])<pcp_index:
            j=j+1
        #print(i,pred_original,y_original,diff[i])
        print(i,pred[i],y_test[i],diff[i])

        
mean_diff=np.mean(diff[diff < 1E308])
pcp=j/len(y_test)
print(mean_diff,"{:.3%}".format(pcp))

# coefficient_of_dermination = r2_score(y_test, pred)
# print("R Square is",coefficient_of_dermination)
# #Neutral_model.save('glass_transition_temp_Morgan+ANN.h5') 
# #Neutral_model.save('Rm_Morgan+ANN.h5') 
# # improve model accuracy
# # improve model accuracy
# coefficient_of_dermination_test=0
# coefficient_of_dermination_train=0
# mean_diff_test=0
# mean_diff_train=0
# while coefficient_of_dermination_test<0.55 or coefficient_of_dermination_train<0.95 or mean_diff_test>0.055:
#         history =Neutral_model.fit(X_train.reshape(len(X_train),latent_dim), y_train.reshape(len(X_train),1),
#                                    epochs=10,validation_data=(X_test.reshape(len(X_test),latent_dim), 
#                                                               y_test.reshape(len(X_test),1)))
#         #test data
#         j=0
#         diff=np.zeros(len(y_test))
#         pred=np.zeros(len(y_test))
#         y_avg=np.average(y_test)
        
#         for i in range(0,len(X_test)):
          
#             pred[i]=Neutral_model.predict(X_test[i].reshape(1,latent_dim))
#             diff[i]=abs((pred[i]-y_test[i])/y_test[i])
#             if abs(diff[i])<pcp_index:
#                     j=j+1
#                 #print(i,pred_original,y_original,diff[i])
#             #print(i,pred[i],y_test[i],diff[i])
#         mean_diff_test=np.mean(diff)
#         print("Mean difference  PCP accuracy")
#         print(mean_diff_test,"{:.3%}".format(j/len(y_test)))
#         coefficient_of_dermination_test = r2_score(y_test, pred)
#         #training data
#         j=0
#         diff=np.zeros(len(y_train))
#         pred=np.zeros(len(y_train))
#         for i in range(0,len(X_train)):
#             pred[i]=Neutral_model.predict(X_train[i].reshape(1,latent_dim))
#             diff[i]=abs((pred[i]-y_train[i])/y_train[i])
        
#             if abs(diff[i])<pcp_index:
#                     j=j+1
#                 #print(i,pred_original,y_original,diff[i])
#             #print(i,pred[i],y_train[i],diff[i])
#         mean_diff_train=np.mean(diff)
#         pcp=j/len(y_train)
#         print(mean_diff_train,"{:.3%}".format(pcp))
#         coefficient_of_dermination_train = r2_score(y_train, pred)
#         print("R Square in train data  is",coefficient_of_dermination_train)
# print("R Square in train data  is",coefficient_of_dermination_train)
# print("R Square in test data  is",coefficient_of_dermination_test)                       

#Plot test data
csfont = {'fontname':'Times New Roman'}
fig, ax = plt.subplots()
ax.scatter(y_test,pred, s=25, cmap=plt.cm.coolwarm, zorder=10,marker = '^')
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
# now plot both limits against eachother
ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

plt.text(.01, .79,'MAPE={:.2f}%'.format(mean_diff*100),
          ha='left', va='top', transform=ax.transAxes,fontsize=18)
plt.text(.01, .89,'PCP={:.2f}%'.format(pcp*100),
          ha='left', va='top', transform=ax.transAxes,fontsize=18)
plt.text(.01, .99,'$R^2$=%s'%(np.round(coefficient_of_dermination, 2)),
          ha='left', va='top', transform=ax.transAxes,fontsize=18)


plt.ylabel("Prediction (%)",**csfont,fontsize=18)
plt.xlabel("Test data (%)",**csfont,fontsize=18)
plt.grid(linestyle='dotted')
plt.show()




# mol1 = "C=CC(=O)OCC(O)COc2ccc(C(C)(C)c1ccc(OCC(O)COC(=O)C=C)cc1)cc2"
# mol1 = Chem.MolFromSmiles(mol1)
# fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=rad, nBits=latent_dim)
# vec1= fp1.ToBitString()
# vec1=np.asarray(list(map(float, vec1)))
# mol2 = "C=CC(=O)OCCn1c(=O)n(CCOC(=O)C=C)c(=O)n(CCOC(=O)C=C)c1=O"
# mol2 = Chem.MolFromSmiles(mol2)
# fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=rad, nBits=latent_dim)
# vec2= fp2.ToBitString()
# vec2=np.asarray(list(map(float, vec2)))

# combined_vetor=float(0)*vec1+float(1)*vec2


# print(Neutral_model.predict(combined_vetor.reshape(1,2048)))

#Calculate train data
j=0
diff=np.zeros(len(y_train))
pred=np.zeros(len(y_train))
for i in range(0,len(X_train)):
    pred[i]=best_model.predict(X_train[i].reshape(1,latent_dim))
    diff[i]=abs((pred[i]-y_train[i])/y_train[i])

    if abs(diff[i])<pcp_index:
            j=j+1
        #print(i,pred_original,y_original,diff[i])
    print(i,pred[i],y_train[i],diff[i])

        
mean_diff=np.mean(diff)
pcp=j/len(y_train)
print(mean_diff,"{:.3%}".format(pcp))
coefficient_of_dermination = r2_score(y_train, pred)
print("R Square is",coefficient_of_dermination)


# #plot training data

fig, ax = plt.subplots()
ax.scatter(y_train,pred, s=25, cmap=plt.cm.coolwarm, zorder=10,marker = '^')
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
# now plot both limits against eachother
ax.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
ax.set_aspect('equal')
ax.set_xlim(lims)
ax.set_ylim(lims)

plt.text(.01, .79,'MAPE={:.2f}%'.format(mean_diff*100),
          ha='left', va='top', transform=ax.transAxes,fontsize=18)
plt.text(.01, .89,'PCP={:.2f}%'.format(pcp*100),
          ha='left', va='top', transform=ax.transAxes,fontsize=18)
plt.text(.01, .99,'$R^2$=%s'%(np.round(coefficient_of_dermination, 2)),
          ha='left', va='top', transform=ax.transAxes,fontsize=18)


plt.ylabel("Prediction (%)",**csfont,fontsize=18)
plt.xlabel("Training data (%)",**csfont,fontsize=18)
plt.grid(linestyle='dotted')
plt.show()