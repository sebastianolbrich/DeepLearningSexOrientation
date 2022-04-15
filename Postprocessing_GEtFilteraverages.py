# -*- coding: utf-8 -*-
"""
Created on Fri May 15 01:08:10 2020

@author: Sebastian
"""

def layer_to_visualize(layer):
   ...:     inputs = [be.learning_phase()] + model.inputs
   ...: 
   ...:     _convout1_f = be.function(inputs, [layer.output])
   ...:     def convout1_f(X):
   ...:         # The [0] is to disable the training phase flag
   ...:         return _convout1_f([0] + [X])
   ...: 
   ...:     convolutions = convout1_f(neu4)  #hier wird das Bild reingegeben
   ...:     convolutions = np.squeeze(convolutions)
   ...: 
   ...:     print ('Shape of conv:', convolutions.shape)
   ...: 
   ...:     n = convolutions.shape[0]
   ...:     n = int(np.ceil(np.sqrt(n)))
   ...:     r=convolutions.shape[2]
   ...: 
   ...:     # Visualization of each filter of the layer
   ...:     fig = plt.figure(figsize=(12,8))
   ...:     for i in range(len(convolutions)):
   ...:         ax = fig.add_subplot(n,n,i+1)
   ...:         ax.imshow(convolutions[i], cmap='gray')
   ...:     for i in range(2):
   ...:       pic=convolutions[:,:,i]
   ...:       pic=np.squeeze(pic)
   ...:       plt.imshow(pic)
   ...:       plt.show()
   ...:     average = np.average(convolutions,axis=2)
   ...:     plt.imshow(average, cmap="gray")
   ...:     plt.show()
   ...:     return average
   ...: # Specify the layer to want to visualize
    
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
    
x1np=np.array(x1)


average = np.average(x1np,axis=0)
plt.imshow(average, cmap="gray")
plt.show()

plt.imshow(average, cmap="jet", alpha = 0.5)
plt.show()