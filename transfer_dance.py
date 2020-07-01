
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers,optimizers,applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import math

X = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

path_train = "F:/Codes/Dance CNN/dataset/train"
path_test = "F:/Codes/Dance CNN/dataset/test"

nb_classes=8
last_block_layer=126
img_width,img_height = 299,299
batch_size = 32
nb_epoch = 15
learn_rate = 1e-4 
momentum = 0.4
transformation_ratio = 0.02

base_model = applications.Xception(input_shape = (img_width,img_height,3), weights = 'imagenet',include_top=False)

#blocking the top model
z = base_model.output
z = layers.GlobalAveragePooling2D()(z)
#z = layers.Dropout(0.5)(z)
predictions = layers.Dense(nb_classes,activation='softmax')(z)

#adding out layer
model = models.Model(base_model.input, predictions)
print(model.summary)

#train only top layers, freeze all the already trained layers
for layer in base_model.layers:
    layer.trainable = False

#doing data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=transformation_ratio,rescale=1./255,zoom_range=transformation_ratio,cval=transformation_ratio,vertical_flip=True,shear_range=transformation_ratio,horizontal_flip=True,width_shift_range=transformation_ratio,height_shift_range=transformation_ratio,fill_mode='nearest', validation_split=0.1) #, validation_split=0.1
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_dataframe(dataframe=X,directory=path_train,x_col="Image",y_col="target",class_mode='categorical',target_size=(img_width,img_height),shuffle=True,batch_size=batch_size,subset='training') #,subset='training'
valid_generator = train_datagen.flow_from_dataframe(dataframe=X,directory=path_train,x_col="Image",y_col="target",class_mode='categorical',target_size=(img_width,img_height),shuffle=True,batch_size=batch_size,subset='validation') #,subset='validation'
test_generator = valid_datagen.flow_from_dataframe(dataframe=X_test,directory=path_test,x_col="Image",y_col=None,class_mode=None,shuffle=False,target_size=(img_width,img_height),batch_size=16)

model.compile(optimizer='nadam',loss='categorical_crossentropy',metrics=['accuracy'])

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID= valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST = math.ceil(test_generator.n/test_generator.batch_size)

model.fit_generator(generator=train_generator,steps_per_epoch=STEP_SIZE_TRAIN,epochs=nb_epoch,validation_data = valid_generator,validation_steps=STEP_SIZE_VALID)

#reloading weights to ensure best epoch is selected
#last layer will be re trained based on train data
for layer in model.layers[:last_block_layer]:
    layer.trainable=False
for layer in model.layers[last_block_layer:]:
    layer.trainable=True

model.compile(optimizer='nadam',loss='categorical_crossentropy',metrics=['accuracy'])

#fine tune model
model.fit_generator(generator=train_generator,steps_per_epoch=STEP_SIZE_TRAIN,epochs=nb_epoch,validation_data =valid_generator,validation_steps=STEP_SIZE_VALID)

test_generator.reset()
predict = model.predict_generator(test_generator, steps=STEP_SIZE_TEST)
predicted_class_indices=np.argmax(predict,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]

results=pd.DataFrame({"Image":X_test.Image, "target":predictions})
results.to_csv("results.csv",index=False)