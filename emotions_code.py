import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
import matplotlib as mpl
from matplotlib import pyplot
import pandas as pd
import sklearn.model_selection 
from sklearn.model_selection import train_test_split
import IPython.display
from IPython.display import clear_output
import time

dat = pd.read_csv("C:\\Users\\SOFI\\Desktop\\project_end2022\\fer2013\\fer2013.csv")
print(dat.head())
print(dat.emotion.unique())

label_to_text = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness',
                 4:'sadness', 5:'surprise', 6:'neutral'}
np_array=np.array(dat.pixels.loc[0].split(' ')).reshape(48,48)
img_array=dat.pixels.apply(lambda x:np.array(x.split(' ')).reshape(48,48).astype('float32'))
img_array=np.stack(img_array,axis=0)

print(img_array.shape)

labels=dat.emotion.values
X_train, X_test, y_train, y_test = train_test_split(img_array, labels, test_size=.1)
X_train=X_train/255
X_test=X_test/255
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
basemodel = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32,(3,3), activation='relu', input_shape=(48,48,1)),
                                       tf.keras.layers.MaxPool2D(2,2),
                                       tf.keras.layers.BatchNormalization(),
                                       
                                       tf.keras.layers.Conv2D(64,(3,3), activation='relu', input_shape=(48,48,1)),
                                       tf.keras.layers.MaxPool2D(2,2),
                                       tf.keras.layers.BatchNormalization(),
                                       
                                       tf.keras.layers.Conv2D(128,(3,3), activation='relu', input_shape=(48,48,1)),
                                       tf.keras.layers.MaxPool2D(2,2),
                                       tf.keras.layers.BatchNormalization(),
                                       
                                       tf.keras.layers.Flatten(),
                                       tf.keras.layers.Dense(128, activation='relu'),
                                       tf.keras.layers.Dense(7, activation='softmax'),
                                       ])

basemodel.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=.0001),
                                                        loss='sparse_categorical_crossentropy',
                                                        metrics=['accuracy'])

checkpoint_path="C:\\Users\\SOFI\\Desktop\\project_end2022"

call_back = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                               monitor='val_accuracy',
                                               verbose=1,
                                               save_freq='epoch',
                                               save_best_only=True,
                                               save_weights_only=False,
                                               mode='max')

basemodel.fit(X_train, y_train, epochs=20,
              validation_split=.1, callbacks=call_back)

final_model=tf.keras.models.load_model(checkpoint_path)
final_model.save("C:\\Users\\SOFI\\Desktop\\project_end2022")

for k in range(40):
    print(f'actual label is {label_to_text[y_test[k]]}')
    predicted_class=final_model.predict(tf.expand_dims(X_test[k],0)).argmax()
    print(f'predicted label is {label_to_text[predicted_class]}')
    pyplot.imshow(X_test[k].reshape((48,48)))
    pyplot.show()
    time.sleep(5)
    clear_output(wait=True)
