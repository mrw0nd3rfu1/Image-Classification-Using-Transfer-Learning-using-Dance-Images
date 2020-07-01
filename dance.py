import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers,optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import math

X = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

path_train = "F:/Codes/Dance CNN/dataset/train"
path_test = "F:/Codes/Dance CNN/dataset/test"

new_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=40,rescale=1/255,zoom_range=0.2,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,horizontal_flip=True,fill_mode='nearest') #, validation_split=0.1
datagen = tf.keras.preprocessing.image.ImageDataGenerator(1/255)
train_generator = new_datagen.flow_from_dataframe(dataframe=X,directory=path_train,x_col="Image",y_col="target",class_mode='categorical',target_size=(256,256),batch_size=16,shuffle=True) #,subset='training'
valid_generator = new_datagen.flow_from_dataframe(dataframe=X,directory=path_train,x_col="Image",y_col="target",class_mode='categorical',target_size=(256,256),batch_size=16,shuffle=False) #,subset='validation'
test_generator = datagen.flow_from_dataframe(dataframe=X_test,directory=path_test,x_col="Image",y_col=None,class_mode=None,shuffle=False,target_size=(256,256),batch_size=16)

models = models.Sequential()
models.add(layers.Conv2D(32, kernel_size=(3,3),strides=(1,1), activation='relu' , input_shape=(256,256,3)))
models.add(layers.MaxPooling2D((2,2)))

models.add(layers.Conv2D(32, kernel_size=(3,3),strides=(1,1), activation='relu'))
models.add(layers.MaxPooling2D((2,2)))
models.add(layers.Dropout(0.5))
models.add(layers.Conv2D(64,  kernel_size=(3,3),strides=(1,1) , activation='relu'))
models.add(layers.MaxPooling2D((2,2)))

#models.add(layers.Conv2D(64,  kernel_size=(2,2), activation='relu'))
models.add(layers.Flatten())
models.add(layers.Dense(64,activation='relu'))
models.add(layers.Dropout(0.5))
#models.add(layers.Dropout(0.25))
models.add(layers.Dense(8,activation='softmax'))

models.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID= valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST = math.ceil(test_generator.n/test_generator.batch_size)

models.fit_generator(generator=train_generator,steps_per_epoch=STEP_SIZE_TRAIN,epochs=50)
#,validation_data=valid_generator,validation_steps=STEP_SIZE_VALID

# test_loss, test_acc = models.evaluate_generator(valid_generator , steps=24)
# print(test_acc)

test_generator.reset()
predict = models.predict_generator(test_generator, steps=STEP_SIZE_TEST)
predicted_class_indices=np.argmax(predict,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

results=pd.DataFrame({"Image":X_test.Image, "target":predictions})
results.to_csv("results.csv",index=False)