import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input, Activation, AveragePooling2D
from keras.layers import add, Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D
from keras.optimizers import SGD, Adagrad, Adam
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

# define parameter
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("mode", "train", "running mode")
tf.app.flags.DEFINE_string("model", 'cnn', "model")
tf.app.flags.DEFINE_string("fig", "0", "whether save training visualization figure")
tf.app.flags.DEFINE_string("save", "0", "whether save the trained model")
tf.app.flags.DEFINE_integer("epoch", 30, "training epochs")
tf.app.flags.DEFINE_float("dropout_rate", 0.2, "")
tf.app.flags.DEFINE_float("learnrate", 0.00003, "")
tf.app.flags.DEFINE_float("val_split", 0.2, "validation split rate")

filter_num_1 = 32
filter_num_2 = 64
dropout_rate = FLAGS.dropout_rate
learnrate = FLAGS.learnrate
epoch = FLAGS.epoch

def load_and_preprocess():
    # define file path
    train_data_path = 'data/train_input.npy'
    train_label_path = 'data/train_label.npy'
    #test_train_path = 'data/test_input.npy'

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

    return x_train[0:360,:,:,:], y_train[0:360], x_train[360:,:,:,:], y_train[360:]

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
	
def Conv2d_BN(x, nb_filter,kernel_size, strides=(1,1), padding='same',name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
 
    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    x = BatchNormalization(axis=3,name=bn_name)(x)
    return x 
def Conv_Block(inpt,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)
        x = add([x,shortcut])
        return x
    else:
        x = add([x,inpt])
        return x
def _res(learnrate):
    inpt = Input(shape=(256,256,3))
    x = ZeroPadding2D((3,3))(inpt)
    x = Conv2d_BN(x,nb_filter=64,kernel_size=(7,7),strides=(2,2),padding='valid')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    #(56,56,64)
    x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=64,kernel_size=(3,3))
    #(28,28,128)
    x = Conv_Block(x,nb_filter=128,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
    x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=128,kernel_size=(3,3))
    #(14,14,256)
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=256,kernel_size=(3,3))
    #(7,7,512)
    x = Conv_Block(x,nb_filter=512,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
    x = Conv_Block(x,nb_filter=512,kernel_size=(3,3))
    x = Conv_Block(x,nb_filter=512,kernel_size=(3,3))
    x = AveragePooling2D(pool_size=(7,7))(x)
    x = Flatten()(x)
    x = Dense(11,activation='softmax')(x)
    model = Model(inputs=inpt,outputs=x)
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

def _savemodel():
    if FLAGS.save != 0:
        model.save(filepath='model.h5')

x_train, y_train, x_val, y_val = load_and_preprocess()
if FLAGS.mode == 'train':
    if FLAGS.model == 'cnn':
        model = _cnn(filter_num_1, filter_num_2, dropout_rate, learnrate)
    elif FLAGS.model == 'res':
        model = _res(learnrate)
    else:
        print("wrong parameter: model!")
        exit()



    # data enhancement
    datagen = ImageDataGenerator(
        rotation_range=20,
        # width_shift_range=0.2,
        # height_shift_range=0.1,
        # zoom_range=0.1,
        # horizontal_flip=True,
        fill_mode='nearest')

    datagen.fit(x_train)
    history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32), validation_data=(x_val, y_val), epochs=epoch, steps_per_epoch=15)
    _savemodel()
elif FLAGS.mode == 'load':
    model = load_model(filepath='model.h5')
    history = model.fit(x=x_train, y=y_train, batch_size=32, validation_split=FLAGS.val_split, epochs=epoch)
    _savemodel()
else:
    print('wrong parameter: mode')
    exit(1)
if FLAGS.fig !=0:
    _visualize()



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