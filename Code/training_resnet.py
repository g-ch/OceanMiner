import keras
from keras.layers import Dense, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import scipy.io as scio
import matplotlib.pyplot as plt

if __name__ == "__main__":
    #load data
    mat_train=u'training_data_bgr_224_224_3.mat'
    train_data=scio.loadmat(mat_train)

    mat_val=u'validation_data_bgr_224_224_3.mat'
    val_data=scio.loadmat(mat_val)

    plt.close('all')
    x_train=train_data['image']
    y_train=train_data['label']
    x_test=val_data['image']
    y_test=val_data['label']

    print 'train original shape:', x_train.shape
    print "data loaded successfully!"

    #set training parameters
    batch_size = 40
    num_classes = 2
    epochs = 30

    # input image dimensions
    img_rows, img_cols = 224, 224

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)
    input_shape = (img_rows, img_cols, 3)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    base_model = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3))

    l = Flatten()(base_model.output)
    l = Dense(128, activation='relu')(l)
    l = Dropout(0.5)(l)
    l = BatchNormalization()(l)
    predictions = Dense(num_classes, activation='softmax')(l)

    # create graph of new model
    model = Model(input=base_model.input, output=predictions)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.summary()

    # training
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('model_224_224_resnet_model.h5')