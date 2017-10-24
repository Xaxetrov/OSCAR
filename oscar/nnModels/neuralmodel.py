from keras.models import Model, load_model, save_model
from keras.layers import Conv2D, Input, Dense, Flatten, BatchNormalization


def get_neural_network():
    try:
        model = load_model("config/mineralshard.knn")
    except OSError:
        sc_i = Input(shape=(64, 64, 2))
        sc_l1 = Conv2D(8, 5, activation='relu', padding='same')(sc_i)
        sc_l1n = BatchNormalization()(sc_l1)
        # reduce screen to 32x32
        sc_l2 = Conv2D(4, 5, strides=(2, 2), activation='relu', padding='same')(sc_l1n)
        sc_l2n = BatchNormalization()(sc_l2)
        sc_f = Flatten()(sc_l2n)
        d1 = Dense(256, activation='relu')(sc_f)
        oa = Dense(2)(d1)
        # output move selection at 16x16
        op = Conv2D(1, 9, strides=(2, 2), padding='same')(sc_l2n)
        # set and compile model
        model = Model(inputs=sc_i, outputs=[oa, op])
        model.compile(optimizer='adam',
                           loss='mean_squared_error')
    return model
