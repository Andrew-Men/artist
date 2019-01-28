import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from numpy import array
from numpy import argmax
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Conv2D, Dense, Activation, MaxPooling2D
from tensorflow.keras.layers import Flatten, BatchNormalization, Dropout
from tensorflow.keras.layers import ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "32", "batch size for training")
tf.flags.DEFINE_string('mode', "train", "Mode train/ predict/ visualize")
tf.flags.DEFINE_bool('load', 'False', "True/ False")

train_data_path = 'data/train_input.npy'
train_label_path = 'data/train_label.npy'
predict_data_path = 'data/test_input.npy'
result_path = "result"
log_path = "log/"
model_name = "log/" + 'my_model.h5'
def split_train(data,label,test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices =shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data[train_indices],data[test_indices],label[train_indices],label[test_indices]

def _cnn(imgs_dim, compile_=True):
    model = Sequential()
    # input: 256x256 images with 3 channels -> (256, 256, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), kernel_initializer='he_normal', activation='relu', input_shape=(256, 256, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), kernel_initializer='he_normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), kernel_initializer='he_normal', activation='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), kernel_initializer='he_normal', activation='relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(256, kernel_initializer='he_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(11, kernel_initializer='glorot_normal', activation='softmax'))
    return model
    
train_input = np.load(train_data_path)
train_label = np.load(train_label_path)

train_data,test_data,train_label,test_label = split_train(train_input, train_label, 0.10)

class_names = ['jmw_turner','george_romney','canaletto',
               'claude_monet','peter_paul_rubens','paul_sandby',
               'rembrandt', 'paul_gauguin', 'john_robert_cozens',
               'richard_wilson','paul_cezanne']

# data preprocess
train_data = train_data / 255.0
test_data = test_data / 255.0
# one hot encode
train_label_encoded = to_categorical(array(train_label))
test_label_encoded = to_categorical(array(test_label))
x_train = train_data
y_train = train_label_encoded
x_test = test_data
y_test = test_label_encoded

if FLAGS.load == True:
    model = load_model(model_name)
else:
    model = _cnn(imgs_dim=(256, 256, 3))
    

    
    
if FLAGS.mode == 'train':    
    adam = Adam(lr=0.0001)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=adam,
        metrics=['accuracy'])    
    model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test) )
    score = model.evaluate(x_test, y_test, batch_size=32)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    #model.save(result_path+'/my_model.h5')  # creates a HDF5 file 'my_model.h5'
#if FLAGS.mode == 'predict':
    predict_data = np.load(predict_data_path)
    result = model.predict(predict_data)
    result_label = argmax(result,axis=1)
    #result_label = [class_names[i] for i in result_label]
    with open(result_path+'/result.txt',"w") as f:
            f.write(str(result_label)) 