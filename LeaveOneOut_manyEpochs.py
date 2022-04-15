
from __future__ import print_function

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras import optimizers

import numpy as np

from keras.callbacks import EarlyStopping

from matplotlib import pyplot as plt

from keras import backend as be

import gc

import pandas as pd

#import tensorflow as tf

#tf.compat.v1.disable_eager_execution()


np.random.seed(1447)  # for reproducibility

batch_size = 100

nb_classes = 2

nb_epoch =50

layer_dict = 1

# input image dimensions

img_rows, img_cols = 16, 2560

# number of convolutional filters to use

nb_filters = 20

# size of pooling area for max pooling

pool_size = (2, 2)

pool_size_small = (1,2)

pool_size_verysmall=(1,1)

# convolution kernel size

kernel_size = (2, 50)



input_norm=True

import h5py 
f = h5py.File('HC_HOMO_HC_ex.mat','r')  
data = f.get('MASTER_HOMO_HETERO') 
data = np.array(data)
X_traintemp=data.transpose()
X_traintemp = X_traintemp[:,0:24,:]
data = f.get('KEY_HOMO_HETERO_0_HOMO') 
data = np.array(data)
y_traintemp=data.transpose()

def shuffle_in_unisono(a, b):   #Shuffle LAbels and Data so the validation after training uses different data
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return (a,b)
shuffle_in_unisono(X_traintemp,y_traintemp)


result = np.zeros((20000)) 
case_results=np.zeros(79)
case_accuracy=np.zeros(79)
case_loss_=np.zeros(79)
wholehistory = []
historylist=[]

for i in range (0,79):
    
    if i<1:
        X_val=X_traintemp[i*124:i*124+124,0:24,0:256]
        Y_val=y_traintemp[i*124:i*124+124,:]
        
        X_train1=X_traintemp[0:i*124,0:24,0:256]
        X_train2=X_traintemp[(i+1)*124:79*124,0:24,0:256]
        X_train= np.concatenate(( X_train1,  X_train2))
        
        Y_train1 =y_traintemp[0:i*124,:]
        Y_train2 =y_traintemp[(i+1)*124:79*124,:]
        Y_train= np.concatenate(( Y_train1,  Y_train2))       
        
        #X_test = data_test[:,2:]
        N = X_train.shape[0]
        M=X_train.shape[2]
        Ntest = X_val.shape[0]
        D = X_train.shape[1]
        R=i*124
        S=i*124+124
 
        print('We have %s observations with %s x %s dimensions'%(N,D,M))
        print('Validation set = %s from sample %s to sample %s'%(i,R,S))
        
        if input_norm:
            mean = np.mean(X_train,axis=0)
            variance = np.var(X_train,axis=0)
            X_train -= mean
            #The 1e-9 avoids dividing by zero
            X_train /= np.sqrt(variance)+1e-9
            X_val -= mean
            X_val /= np.sqrt(variance)+1e-9
           # X_test -= mean
           # X_test /= np.sqrt(variance)+1e-9
                           

    else:
        
       # X_train =np.zeros((9672,24,256,1))
       # X_val =  np.zeros((124,24,256,1))
        X_val=X_traintemp[i*124:i*124+124,0:24,0:256]
        Y_val=y_traintemp[i*124:i*124+124]
        
        
        #X_train1=np.zeros((i*124,24,256,1))
        #X_train2=np.zeros((79*124-(i+1)*124,24,256,1))
        
        X_train1=X_traintemp[0:i*124,0:24,0:256]
        X_train2=X_traintemp[(i+1)*124:79*124,0:24,0:256]
        X_train= np.concatenate(( X_train1,  X_train2))
        
        
        #Y_train1 =np.zeros((i*124,2))
        #Y_train2 =np.zeros((79*124-(i+1)*124,2))
        
        #Y_train = np.zeros((124,24,256,1))
        #Y_val = np.zeros((124,2))
        Y_train1 =y_traintemp[0:i*124,:]
        Y_train2 =y_traintemp[(i+1)*124:79*124,:]
        Y_train= np.concatenate(( Y_train1,  Y_train2))
        
        N = X_train.shape[0]
        M=X_train.shape[2]
        Ntest = X_val.shape[0]
        D = X_train.shape[1]
        
        #y_test = data_test[:,0:2]
        print('We have %s observations with %s x %s dimensions'%(N,D,M))
        print('Validation set = Nr %s from sample %s to sample %s'%(i,i*124,i*124+124))
        
        if input_norm:
            mean = np.mean(X_train,axis=0)
            variance = np.var(X_train,axis=0)
            X_train -= mean
            #The 1e-9 avoids dividing by zero
            X_train /= np.sqrt(variance)+1e-9
            X_val -= mean
            X_val /= np.sqrt(variance)+1e-9
           # X_test -= mean
           # X_test /= np.sqrt(variance)+1e-9
        
 
        
   
    
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_val.shape[0], 'Testing samples from one Leave Out Subject')

    X_train = X_train.reshape(X_train.shape[0], 24, 256, 1)
    X_val = X_val.reshape(X_val.shape[0], 24, 256, 1)
    X_traintemp = X_traintemp.reshape(X_traintemp.shape[0],24,256,1)
    input_shape = (24, 256, 1)  

    print('X_train reshape:', X_train.shape)
    print(X_train.shape[0], 'reshape train samples')
    print(X_val.shape[0], 'reshape validation samples')  
        
    cb = EarlyStopping(monitor='val_loss', patience=4, verbose = 1)
    cb_list = [cb]    
   
    be.clear_session()   
   # del model 
    model = Sequential()
    
        
    model.add(Convolution2D(100, 2, 2,
    
                            border_mode='valid',
    
                            input_shape=input_shape, name = "erstes"))
    
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=pool_size))
    
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(200, 2, 2, name = "zweites"))
    
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=pool_size))
    
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(300, 2, 2, name = "drittes"))
    
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=pool_size))
    
    model.add(Dropout(0.25))
     
    model.add(Convolution2D(200, 2, 4, name="viertes"))
    
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=pool_size_small))
    
    model.add(Dropout(0.25))  
    
    model.add(Convolution2D(100, 1, 4, name="funftes"))
    
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=pool_size_verysmall))
    
    model.add(Dropout(0.25))
        
    model.add(Convolution2D(100, 1, 4, name="sechstes"))
    
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(pool_size=pool_size_verysmall))
    
    #model.add(Dropout(0.25))
        
    model.add(Flatten())
    
    model.add(Dense(10000))
    
    model.add(Activation('relu'))
    
    model.add(Dropout(0.5))
    
    model.add(Dense(nb_classes))
    
    model.add(Activation('softmax'))
    
    op=optimizers.Adamax(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #op = optimizers.Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    model.compile(loss='categorical_crossentropy',
    
                  optimizer=op,
    
                  metrics=['accuracy'])
    
    #model.save_weights('model.h5')
    #model.load_weights('model.h5')

  
    if i==0:
      model.summary()
    
    print('Schleife Nummer:', i)
    
  #  history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
    
  #            verbose=1, validation_data=(X_val, Y_val), shuffle=True, callbacks=cb_list)
    
    history = model.fit(X_train, Y_train,validation_split=0.1, batch_size=batch_size, epochs=nb_epoch,
    
            verbose=1, shuffle=True, callbacks=cb_list)   
    

   
    
    score = model.evaluate(X_val,Y_val, verbose=1)
        
    print('Test Validation Loss for whole leave out batch sample:', score[0])
        
    print('Test Accuracy for whole leave out batch sample:', score[1])

    wholehistory.append(history)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    
    if score[1]>0.5:
            
          onehot=1
    else:
          onehot=0
          
    
    case_results[i]=onehot
    case_accuracy[i]=score[1]
    case_los=score[0]
    if i<78:

      be.clear_session()   
      del model
      gc.collect()
      del X_train,X_val
     # tf.compat.v1.reset_default_graph() 
    
    history_df = pd.DataFrame(history.history)
    historyscore = pd.DataFrame(score)
    
   
    if i==0:
        history_df.to_csv('historyAll.csv', index=False)
        historyscore.to_csv('historyScore.csv', index=False)
        
    else:
        historylist=pd.read_csv('historyAll.csv')
        historylistscore=pd.read_csv('historyScore.csv')
        
        historylist=historylist.append(history_df)
        historylistscore=historylistscore.append(historyscore)
    
        historylist.to_csv('historyAll.csv', index=False)
        historylistscore.to_csv('historyScore.csv', index=False)
    
 
#print (historylistscore)     
finale=sum(case_results)/79*100
print ('finales Ergebniss: richtige Ergebnisse in Prozent:', finale)


