import tensorflow as tf
from keras.layers import (Input, Conv2D, Concatenate, GlobalAveragePooling2D,
                                     Dense, MaxPooling2D, Dropout)
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model


class Chemception():
    def __init__(self, dense_layers=1, neurons=512, dropout=0.3, img_size = 80, config=None):
        
        if config:
            self.neurons = config['neurons']
            self.dense_layers = config['dense_layers']
            self.input_shape = (config['img_size'], config['img_size'], 3)
            self.dropout = config['dropout']
        else:
            self.neurons = neurons
            self.dense_layers = dense_layers
            self.input_shape = (img_size,img_size,3)
            self.dropout = dropout

    def build(self):
        # Load the InceptionV3 model, excluding the top layers
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=self.input_shape)
        # Extract the output of the final layer
        x = base_model.output
        # Apply Global Average Pooling ( according to arch )
        x = GlobalAveragePooling2D()(x)
        # Softmax
        # for i in range(self.dense_layers):
        #     x = Dense(self.neurons/(i+1), activation='relu')(x)
        #     x = Dropout(self.dropout)(x)
        x = Dense(1024, activation='relu')(x)

        # A binary classifier (sigmoid)
        predictions = Dense(1, activation='softmax')(x)
        
        # This is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        
        for layer in base_model.layers:
            layer.trainable = False

            
        return model