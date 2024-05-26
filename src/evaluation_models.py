import tensorflow as tf
from keras import layers, models, optimizers, callbacks, utils
from pathlib import Path
import math
import input_reader as ipr
import metrics as met
import numpy as np
import os

tf.random.set_seed(2)

# Device setting
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

physical_devices = tf.config.list_physical_devices('GPU') 
for gpu_instance in physical_devices: 
    tf.config.experimental.set_memory_growth(gpu_instance, True)

def DeepSplicer(length):

    model = models.Sequential(name='DeepSplicer')

    model.add(layers.Conv1D(filters=50, kernel_size=9, strides=1, padding='same', batch_input_shape=(None, length, 4), activation='relu'))
    model.add(layers.Conv1D(filters=50, kernel_size=9, strides=1, padding='same', activation='relu'))
    model.add(layers.Conv1D(filters=50, kernel_size=9, strides=1, padding='same', activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(100,activation='relu'))
    
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(2,activation='softmax'))
    
    adam = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def IntSplice(input_number):

    model = models.Sequential(name='IntSplice')

    layer1 = layers.Conv1D(filters=64, kernel_size=10, strides=4, padding='same', batch_input_shape=(None, input_number, 4), activation='relu')
    layer2 = layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu')
    layer3 = layers.Conv1D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu')
    layer4 = layers.Conv1D(filters=512, kernel_size=2, strides=1, padding='same', activation='relu')

    model.add(layer1)
    model.add(layers.MaxPool1D(pool_size=2))
    model.add(layers.Dropout(0.3))

    model.add(layer2)
    model.add(layers.MaxPool1D(pool_size=2))
    model.add(layers.Dropout(0.2))

    model.add(layer3)
    model.add(layers.MaxPool1D(pool_size=2))
    model.add(layers.Dropout(0.3))

    model.add(layer4)
    model.add(layers.MaxPool1D(pool_size=2))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation='relu', name='layer_dense')) # 20
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(2, activation='softmax', name='out'))

    model.summary()

    adam = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])  # categorical
    
    return model

def SpliceFinder(length):

    model = models.Sequential(name='SpliceFinder')
        
    model.add(layers.Conv1D(filters=50, kernel_size=9, strides=1, padding='same', batch_input_shape=(None, length, 4), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(100,activation='relu'))
    
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(2,activation='softmax'))
    
    adam = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def spliceRover(input_number):
    
    model = models.Sequential()

    layer1 = layers.Conv1D(filters=70, kernel_size=9, strides=1, padding='same', batch_input_shape=(None, input_number, 4), activation='relu')
    layer2 = layers.Conv1D(filters=100, kernel_size=7, strides=1, padding='same', activation='relu')
    layer3 = layers.Conv1D(filters=100, kernel_size=7, strides=1, padding='same', activation='relu')
    layer4 = layers.Conv1D(filters=200, kernel_size=7, strides=1, padding='same', activation='relu')
    layer5 = layers.Conv1D(filters=250, kernel_size=7, strides=1, padding='same', activation='relu')

    model.add(layer1)
    model.add(layers.Dropout(0.2))
    model.add(layer2)
    model.add(layers.Dropout(0.2))
    model.add(layer3)
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Dropout(0.2))
    model.add(layer4)
    model.add(layers.MaxPooling1D(4))
    model.add(layers.Dropout(0.2))
    model.add(layer5)
    model.add(layers.MaxPooling1D(4))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2, activation='softmax'))

    model.summary()

    sgd = optimizers.SGD(learning_rate=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    return model

def scheduler(epoch):
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 5
    learning_rate = initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    
    return learning_rate

def training_process(model, x_train, y_train, x_valid, y_valid, x_test, y_test):
    my_callbacks = [callbacks.EarlyStopping(patience=2)]

    history = model.fit(x_train, y_train, epochs=20, batch_size=64, 
                        validation_data=(x_valid, y_valid), callbacks=my_callbacks)

    loss, accuracy = model.evaluate(x_test, y_test)

    print('testing loss: {}'.format(loss))
    print('testing accuracy: {}'.format(accuracy))

    return history
    
# positive real data & negative real data
trainX, trainY, validX, validY, testX, testY = ipr.readInputs("/home/jiwon/BScProject/evaluation_models/GWH_acceptors_chr1_pos.txt", "/home/jiwon/BScProject/evaluation_models/GWH_acceptors_chr1_neg.txt")

# positive synthetic data & negative real data
trainX, trainY, validX, validY, testX2, testY2 = ipr.readInputs("/home/jiwon/BScProject/evaluation_models/Synthetic_Sequences_Epoch100_wholdout.txt", "/home/jiwon/BScProject/evaluation_models/GWH_acceptors_chr1_neg.txt")

# Load the model
model = spliceRover(trainX.shape[1])

# Training
print('----------------Start Training----------------')
history = training_process(model, trainX, trainY, validX, validY, testX, testY)
model.save('/home/jiwon/BScProject/evaluation_models/spliceRover_trained_syn.h5')

# Evaluation
preds = model.predict(testX).astype('float')

m = met.metric(preds, testY)
print('Recall =', m.recall())
print('Precision =', m.precision())
print('F1-Score =', m.f1score())
print('MCC =', m.mcc())