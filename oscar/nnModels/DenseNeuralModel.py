from keras.models import Model, load_model, save_model
from keras.layers import Conv2D, Input, Dense, Flatten, BatchNormalization, Concatenate, Reshape


def get_neural_network():
    try:
        model = load_model("Neuralnetwork/DenseMineralShard.knn")
    except OSError:
        sc_i = Input(shape=(1, 64, 64, 2), name='Input')
        sc_if = Reshape(target_shape=(64, 64, 2), name='InputReshape')(sc_i)
        sc_l1 = Conv2D(8, 5, activation='relu', padding='same', name='Convolution1')(sc_if)
        sc_l1n = BatchNormalization(name='NormalizationC1')(sc_l1)
        # reduce screen to 32x32
        sc_l2 = Conv2D(4, 5, strides=(2, 2), activation='relu', padding='same', name='Convolution2')(sc_l1n)
        sc_l2n = BatchNormalization(name='NormalizationC2')(sc_l2)
        # sc_f = Flatten(name='FlattenC2')(sc_l2n)
        # d1 = Dense(64, activation='relu', name='DenseNonSpacial')(sc_f)
        # oa = Dense(2, name='DenseActionOutput')(d1)
        # output move selection at 16x16
        op = Conv2D(1, 5, strides=(2, 2), padding='same', name='ConvolutionSpacialActionOutput')(sc_l2n)
        opf = Flatten(name='FlattenSpacialAction')(op)
        # o = Concatenate(axis=1, name='ConcatenateOutput')([oa, opf])
        # set and compile model
        model = Model(inputs=sc_i, outputs=opf)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy')
    return model
