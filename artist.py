import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD, Adagrad, Adam
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

# define parameter
dropout_rate = 0.2
filter_num_1 = 16
filter_num_2 = 32
learnrate = 0.0001

# define file path
train_data_path = 'data/train_input.npy'
train_label_path = 'data/train_label.npy'
test_train_path = 'data/test_input.npy'

# load data
train_input = np.load(train_data_path)
train_label = np.load(train_label_path)

# split and preprocess data
# class_names = ['jmw_turner', 'george_romney', 'canaletto',
#                'claude_monet', 'peter_paul_rubens', 'paul_sandby',
#                'rembrandt', 'paul_gauguin', 'john_robert_cozens',
#                'richard_wilson', 'paul_cezanne']
train_data = train_input / 255.0
train_label_encoded = to_categorical(np.array(train_label))

# Generate dummy data
x_train = train_data
y_train = train_label_encoded
index=np.arange(len(y_train))
np.random.shuffle(index)
 
x_train=x_train[index,:,:,:]#X_train是训练集，y_train是训练标签
y_train=y_train[index]

# define network structure
def _cnn(filter_num_1, filter_num_2, dropout_rate, learnrate):
    model = Sequential()
    model.add(Conv2D(filter_num_1, (3, 3), activation='relu', input_shape=(256, 256, 3)))
    model.add(Conv2D(filter_num_1, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Conv2D(filter_num_2, (3, 3), activation='relu'))
    model.add(Conv2D(filter_num_2, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(11, activation='softmax'))

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #adagrad = Adagrad(lr=0.0006, epsilon=None, decay=0.0)
    adam = Adam(lr=learnrate)# 0.0001

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])
    return model

# plot training process
def _visualize():
    # FIX CRASH #
    import matplotlib
    matplotlib.use("TkAgg")
    # --------- #
    
    from matplotlib import pyplot as plt
    plt.figure(1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('train.png')

model = _cnn(filter_num_1, filter_num_2, dropout_rate, learnrate)
history = model.fit(x=x_train, y=y_train, batch_size=32, validation_split=0.2, epochs=100)

# model.save(filepath='/Users/eis/Desktop/data/model-bn.h5')



# data enhancement
# datagen = ImageDataGenerator(
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest')

# datagen.fit(x_train)