from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import *
import pydot
import os
class Custom_Conv_LSTM(tf.keras.Model):
    
    def __init__(self):
        super(Custom_Conv_LSTM, self).__init__()
        self.layer1 = Sequential(
            [keras.Input(shape=(250, 12)),
            LSTM(128,return_sequences=True),
            LSTM(256,return_sequences=True),
            LSTM(512),
            Flatten()
            ]
        )
        self.layer2 = Sequential(
            [   keras.layers.Input(shape=(250, 12)),
                keras.layers.Conv1D(filters=128, kernel_size=15,input_shape=(250, 12), padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.Activation(activation='relu'),
                keras.layers.SpatialDropout1D(0.1),

                keras.layers.Conv1D(filters=256, kernel_size=10, padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu'),
                keras.layers.SpatialDropout1D(0.1),

                keras.layers.Conv1D(512, kernel_size=5,padding='same'),
                keras.layers.BatchNormalization(),
                keras.layers.Activation('relu'),
                keras.layers.Dropout(0.2),
                Flatten()
            ]
            
        )
        
        self.layer3 = Sequential(
            [   
            #  keras.layers.Input(shape=(X_train.shape[0], 512+128000 )),
             Dense(2,activation='softmax')
            ]
            
        )
    
    def call(self, x):
        
        x1 = self.layer1(x)
        print("x1.shape : ",x1.shape)
        x2 = self.layer2(x)
        print("x2.shape : ",x2.shape)
        x3 = keras.layers.concatenate([x1,x2])
        x3 = self.layer3(x3)
        return x3   
    
model = Custom_Conv_LSTM()  
model.build(input_shape=( 100, 250,12))
model.summary()
if not os.path.isfile('Model_Diagram.png'):
        keras.utils.plot_model(model, to_file='Model_Diagram.png', show_shapes=True, show_layer_names=True, expand_nested=True)