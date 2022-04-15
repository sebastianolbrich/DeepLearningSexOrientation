# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 15:09:35 2020

@author: Sebastian
"""

def layer_to_visualize(layer):
        inputs = [be.learning_phase()] + model.inputs
    
        _convout1_f = be.function(inputs, [layer.output])
        def convout1_f(X):
            # The [0] is to disable the training phase flag
            return _convout1_f([0] + [X])
    
        convolutions = convout1_f(neu4)  #hier wird das Bild reingegeben
        convolutions = np.squeeze(convolutions)
        
        if convolutions.ndim==2:
   #    if convolutions.ndim==2:
            convolutions = convolutions.reshape(1, convolutions.shape[0],convolutions.shape[1])
    
        print ('Shape of conv:', convolutions.shape)
    
        n = convolutions.shape[0]
        n = int(np.ceil(np.sqrt(n)))
        r=convolutions.shape[2]
    
        # Visualization of each filter of the layer
       # fig = plt.figure(figsize=(12,8))
       # for i in range(len(convolutions)):
        #    ax = fig.add_subplot(n,n,i+1)
        #    ax.imshow(convolutions[i], cmap='gray')
       # for i in range(2):
        #  pic=convolutions[:,:,i]
         # pic=np.squeeze(pic)
        #  plt.imshow(pic)
        #  plt.show()
        average = np.average(convolutions,axis=2)
       # plt.imshow(average, cmap="gray")
        plt.show()
        return average
    # Specify the layer to want to visualize
    
x1=[[]]*124
x2=[[]]*124
x3=[[]]*124
x4=[[]]*124
x5=[[]]*124
x6=[[]]*124
x7=[[]]*124
x8=[[]]*124
x9=[[]]*124
x10=[[]]*124
x11=[[]]*124
x12=[[]]*124
x13=[[]]*124
x14=[[]]*124
x15=[[]]*124
x16=[[]]*124
x17=[[]]*124
x18=[[]]*124
x19=[[]]*124
x20=[[]]*124
x21=[[]]*124
x22=[[]]*124
x23=[[]]*124
x24=[[]]*124
x25=[[]]*124
x26=[[]]*124
x27=[[]]*124
x28=[[]]*124
x29=[[]]*124


for i in range (0,124):
    _img = X_train[i,0:24,0:256]
    neu = np.squeeze(_img, axis=2)
    neu4 = _img.reshape(1, 24, 256, 1)
    x1[i]=layer_to_visualize(model.layers[1])
    x2[i]=layer_to_visualize(model.layers[2])
    x3[i]=layer_to_visualize(model.layers[3])
    x4[i]=layer_to_visualize(model.layers[4])
    x5[i]=layer_to_visualize(model.layers[5])
    x6[i]=layer_to_visualize(model.layers[6])
    x7[i]=layer_to_visualize(model.layers[7])
    x8[i]=layer_to_visualize(model.layers[8])
    x9[i]=layer_to_visualize(model.layers[9])
    x10[i]=layer_to_visualize(model.layers[10])
    x11[i]=layer_to_visualize(model.layers[11])
    x12[i]=layer_to_visualize(model.layers[12])
    x13[i]=layer_to_visualize(model.layers[13])
    x14[i]=layer_to_visualize(model.layers[14])
    x15[i]=layer_to_visualize(model.layers[15])
    x16[i]=layer_to_visualize(model.layers[16])
    x17[i]=layer_to_visualize(model.layers[17])
    x18[i]=layer_to_visualize(model.layers[18])
    x19[i]=layer_to_visualize(model.layers[19])
    x20[i]=layer_to_visualize(model.layers[20])
    x21[i]=layer_to_visualize(model.layers[21])
    x22[i]=layer_to_visualize(model.layers[22])
   

    

plt.show()
x1np.shape
x1np=np.array(x1)
x2np=np.array(x2)
x3np=np.array(x3)
x4np=np.array(x4)
x5np=np.array(x5)
x6np=np.array(x6)
x7np=np.array(x7)
x8np=np.array(x8)
x9np=np.array(x9)
x10np=np.array(x10)
x11np=np.array(x11)
x12np=np.array(x12)
x13np=np.array(x13)
x14np=np.array(x14)
x15np=np.array(x15)
x16np=np.array(x16)
x17np=np.array(x17)
x18np=np.array(x18)
x19np=np.array(x19)
x20np=np.array(x20)
x21np=np.array(x21)
x22np=np.array(x22)




average = np.average(x1np,axis=0)
plt.imsave('1.png', image)
plt.imshow(average, cmap="jet", alpha = 0.5)
plt.show()

average = np.average(x12np,axis=0)
plt.imshow(average, cmap="gray")
plt.show()

plt.imshow(average, cmap="jet", alpha = 0.5)
plt.show()

average = np.average(x13np,axis=0)
plt.imshow(average, cmap="gray")
plt.show()

plt.imshow(average, cmap="jet", alpha = 0.5)
plt.show()

average = np.average(x14np,axis=0)
plt.imshow(average, cmap="gray")
plt.show()

plt.imshow(average, cmap="jet", alpha = 0.5)
plt.show()

average = np.average(x15np,axis=0)
plt.imshow(average, cmap="gray")
plt.show()

plt.imshow(average, cmap="jet", alpha = 0.5)
plt.show()

average = np.average(x16np,axis=0)
plt.imshow(average, cmap="gray")
plt.show()
plt.imshow(average, cmap="jet", alpha = 0.5)
plt.show()

average = np.average(x17np,axis=0)
plt.imshow(average, cmap="gray")
plt.show()
plt.imshow(average, cmap="jet", alpha = 0.5)
plt.show()

average = np.average(x18np,axis=0)
plt.imshow(average, cmap="gray")
plt.show()
plt.imshow(average, cmap="jet", alpha = 0.5)
plt.show()

average = np.average(x19np,axis=0)
plt.imshow(average, cmap="gray")
plt.show()
plt.imshow(average, cmap="jet", alpha = 0.5)
plt.show()

average = np.average(x29np,axis=0)
plt.imshow(average, cmap="gray")
plt.show()
plt.imshow(average, cmap="jet", alpha = 0.5)
plt.show()


pic3_filter19=x19np[2,:,:]
fig19=plt.imshow(pic3_filter19, cmap="jet", alpha = 0.5, aspect='auto')