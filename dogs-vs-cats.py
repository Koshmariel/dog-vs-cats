#Importing the Kerasl libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
import os
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

#image dimensions
image_width = 128
image_height =128
image_channels =3    # 3channels for color images
image_size = (image_width, image_height)

#Building th CNN
classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), padding ='same', input_shape=(image_width, image_height, image_channels), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Dropout(0.25))

classifier.add(Flatten())
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.25))
classifier.add(Dense(units = 2, activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])





#Early Stop
#to prevent over fitting stop the learning after 10 epochs if val_loss value is not decreasing
earlystop = EarlyStopping(patience=10,
                          verbose=1,
                          restore_best_weights=True)

#Learning Rate Reduction
#reduce the LR when accucarcy will not increase for 2 steps
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=2,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)

callbacks = [earlystop, learning_rate_reduction]



#Prepairing the data
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rotation_range=10,
                                   rescale=1./255,
                                   shear_range=0.1, #rhombus
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/train_set',
                                                target_size=(image_width, image_height), #same as input
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(image_width, image_height),
                                            batch_size=32,
                                            class_mode='binary')

#############################
# Show sample image

def findFilesInFolder(path, pathList, extension, subFolders = True):
    import os
    """  Recursive function to find all files of an extension type in a folder (and optionally in all subfolders too)

    path:        Base directory to find files
    pathList:    A list that stores all paths
    extension:   File extension to find
    subFolders:  Bool.  If True, find files in all subfolders under path. If False, only searches files in the specified folder
    """

    try:   # Trapping a OSError:  File permissions problem I believe
        for entry in os.scandir(path):
            if entry.is_file() and entry.path.endswith(extension):
                pathList.append(entry.path)
            elif entry.is_dir() and subFolders:   # if its a directory, then repeat process as a nested function
                pathList = findFilesInFolder(entry.path, pathList, extension, subFolders)
    except OSError:
        print('Cannot access ' + path +'. Probably a permissions error')

    return pathList

import random
from keras.preprocessing.image import load_img

dir_name=os.getcwd()
extension='.jpg'
pathList = []
pathList = findFilesInFolder(dir_name, pathList, extension, True)
sample = random.choice(pathList)
image = load_img(sample)  
plt.suptitle('Original image')
plt.imshow(image)
plt.show()


#############################
# Show ImageDataGenerator results


# Creating a dataset which contains just one image.
from numpy import asarray
image = load_img(sample, target_size=(128,128))
image=asarray(image)
images = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) #image must be as array
plt.suptitle('Resized image')
plt.imshow(images[0])
plt.show()

train_datagen.fit(images)
image_iterator = train_datagen.flow(images)


# Plot the images given by the iterator
plt.rcParams["figure.figsize"] = (18,9)
fig, axis = plt.subplots(nrows=3, ncols=4)
for ax in axis.flat:
    ax.imshow(image_iterator.next()[0].astype('float32'))
#plt.tight_layout()
plt.suptitle('ImageDataGenerator results')
plt.show()



#############################
#Fitting the CNN to the images
epochs_num = 30 #set separate to test different models


history = classifier.fit_generator(training_set,
                                   epochs=epochs_num,
                                   validation_data=test_set,
                                   callbacks=callbacks)


#############################


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


epochs = range(1, len(acc) + 1)

plt.rcParams["figure.figsize"] = (18,9)
#drawing Training and Test loss
plt.plot(epochs, loss, 'b', label='Train loss')
plt.plot(epochs, val_loss, 'b--', label='Test loss')


plt.title('Training and Test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(fontsize='medium')

plt.show()

plt.clf()   # clear figure


#drawing Training and Test accuracy
plt.plot(epochs, acc, 'b', label='Train acc')
plt.plot(epochs, val_acc, 'b--', label='Test acc')

plt.title('Training and Test accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
#plt.ylim(0, 80)
plt.legend(fontsize='medium')

plt.show()

plt.clf()   # clear figure


#classifier.save('model')
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
#tf.compat.v1.enable_eager_execution()
from tensorflow.keras.models import load_model
classifier=load_model('model')

#Take an examplse from the input directories and classify it
import os
import random
from keras.preprocessing.image import load_img
import matplotlib.pyplot as plt
from numpy import asarray
dir_name=os.getcwd()
extension='.jpg'
pathList = []
pathList = findFilesInFolder(dir_name, pathList, extension, True)
sample = random.choice(pathList)
image = load_img(sample, target_size=(128,128))
plt.imshow(image)
plt.show()
image=asarray(image) #transform image into array of (shape width, height, 3)
images = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])) #add dimension to transform array into a batch



#import tensorflow as tf

y_example=classifier.predict(images)

if y_example[0,0]==1.0:
    print("It's a cat")
elif y_example[0,1]==1.0:
    print("It's a dog")
else:
    print("Something is wrong")
    
import numpy as np
#Neuron corresponding to the predicted class
neuron_num = np.argmax(y_example[0])

# Prediction entry in the prediction vector
model_output = classifier.output[:, neuron_num]

# The output feature map of the the last convolutional layer in the model
#classifier.summary()
last_conv_layer = classifier.get_layer('conv2d_3')

# The gradient of the predicted class with regard to  the output feature map of `dense_2`
from tensorflow.keras import backend as K
grads = K.gradients(model_output, last_conv_layer.output)[0]

# This is a vector of shape (512,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads = K.mean(grads, axis=(0, 1, 2))

# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate = K.function([classifier.input], [pooled_grads, last_conv_layer.output[0]])

# These are the values of these two quantities, as Numpy arrays, given our sample image
pooled_grads_value, conv_layer_output_value = iterate([images])


# The channel-wise mean of the resulting feature map is our heatmap of class activation
heatmap = np.mean(conv_layer_output_value, axis=-1)


heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()


import cv2

# loading the original image
img = cv2.imread(sample)

# Resizing the heatmap to have the same size as the original image
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

# Converting the heatmap to RGB
heatmap = np.uint8(255 * heatmap)

# Applying the heatmap to the original image
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 0.4 is a heatmap intensity factor
superimposed_img = heatmap * 0.4 + img


cv2.imwrite('heatmap.jpg', superimposed_img)