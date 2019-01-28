from sklearn.model_selection import train_test_split
import sklearn
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD, Adagrad
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

# FIX CRASH #
import matplotlib
matplotlib.use("TkAgg")
# --------- #
from matplotlib import pyplot as plt

# define parameter
dropout_rate = 0.3
filter_num_1 = 16
filter_num_2 = 32

# define file path
train_data_path = 'data/train_input.npy'
train_label_path = 'data/train_label.npy'
test_train_path = 'data/test_input.npy'

# load data
train_input = np.load(train_data_path)
train_label = np.load(train_label_path)

# split and preprocess data
train_data, test_data, train_label, test_label = train_test_split(train_input, train_label, test_size=0.10,
                                                                  random_state=0)
# class_names = ['jmw_turner', 'george_romney', 'canaletto',
#                'claude_monet', 'peter_paul_rubens', 'paul_sandby',
#                'rembrandt', 'paul_gauguin', 'john_robert_cozens',
#                'richard_wilson', 'paul_cezanne']
train_data = train_data / 255.0
test_data = test_data / 255.0
train_label_encoded = to_categorical(np.array(train_label))
test_label_encoded = to_categorical(np.array(test_label))

# Generate dummy data
x_train = train_data
y_train = train_label_encoded
x_test = test_data
y_test = test_label_encoded

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

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adagrad = Adagrad(lr=0.001, epsilon=None, decay=0.0)

model.compile(loss='categorical_crossentropy', optimizer=adagrad, metrics=['acc'])

# data enhancement
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

datagen.fit(x_train)

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                              steps_per_epoch=len(x_train) / 32, epochs=50, validation_data=(x_test, y_test))

fit(x_train, y_train, batch_size=32, epochs=80, validation_data=(x_test, y_test))

model.save(filepath='/Users/eis/Desktop/data/model-bn.h5')

# plot training process
plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# plt.figure(2)
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


score = model.evaluate(x_test, y_test, batch_size=32)
print(score)

model = load_model('/Users/eis/Desktop/data/model-bn.h5')
pre_label = model.predict(test_data)

label = []
for i in pre_label:
    label += np.where(i == np.max(i))[0].tolist()

print(pre_label)
print(label)
