import math
from keras.models import Model, load_model, save_model
from keras.layers import Conv2D, Input, Dense, Flatten, \
    BatchNormalization, Reshape, Activation, Permute, MaxPooling2D

from learning_tools.A3C_learner.constants import *

current_neural_network_file = "learning_tools/learning_nn/" + ENV + ".knn"


def get_neural_network(input_shape, output_shape,
                       file_path=current_neural_network_file,
                       loss='mean_squared_error'):
    """
    Constructs a NN model, either by loading it from a file or by creating it from scratch
    :param input_shape: Shape of the input given to the NN
    :param output_shape: Shape of the output retrieved from the NN
    :param file_path: Path to the file that contains maybe
    :param loss: a loss function or function_name
    :return: the built neural_network model
    """
    global current_neural_network_file
    current_neural_network_file = file_path
    try:
        model = load_model(current_neural_network_file)
        print("old NN charged from", file_path)
    except OSError:
        # check output size
        for out_size in output_shape:
            assert type(out_size) == int
        # check input size
        print(input_shape)
        assert len(input_shape) > 3 and input_shape[-1] == 64 and input_shape[-2] == 64

        sc_i = Input(batch_shape=input_shape,
                     name='input')
        # keep only the three last dimensions
        sc_if = Reshape(target_shape=(input_shape[-3], input_shape[-2], input_shape[-1]),
                        name='InputReshape'
                        )(sc_i)
        # set the layers as the last dimension
        sc_ip = Permute((2, 3, 1),
                        name="PermuteDimensions"
                        )(sc_if)
        # first screen layer, reduce it to 32x32 with 32 filters (layers)
        sc_l1 = Conv2D(32,
                       5,
                       strides=(2, 2),
                       activation='relu',
                       padding='same',
                       name='conv2d_layer1'
                       )(sc_ip)
        # sc_l1n = BatchNormalization(name='normalization1')(sc_l1)
        # reduce screen to 16x16 with 8 filters
        sc_l2 = Conv2D(8,
                       5,
                       strides=(2, 2),
                       activation='relu',
                       padding='same',
                       name='conv2d_layer2'
                       )(sc_l1)
        # sc_l2n = BatchNormalization(name='normalization2')(sc_l2)
        # reduce action space before Dense layer (4x4)
        sc_p = MaxPooling2D(pool_size=(4, 4))(sc_l2)
        sc_f = Flatten(name='flatten_spacial')(sc_p)
        d1 = Dense(8,
                   activation='relu',
                   name='first_layer_for_none_spacial'
                   )(sc_f)

        # generate output shapes
        output_layers = []
        for i, size in enumerate(output_shape):
            if size == 1:
                # this is value output, isn't it ?
                value = Dense(1,
                              activation='linear',
                              name='value' + str(i)
                              )(d1)
                output_layers.append(value)
            elif size % 256 == 0:
                number_of_output_layer = size // 256
                # this is a spacial action with 16*16 output
                spacial = Conv2D(number_of_output_layer,
                                 5,
                                 # strides=(2, 2),
                                 padding='same',
                                 activation='relu',
                                 name='spacial_policy_' + str(i)
                                 )(sc_l2)
                # set the layers as the first dimension
                out = Permute((3, 1, 2), name='permute_dimension_out_' + str(i))(spacial)
                # flatten the output
                out = Reshape(target_shape=(-1,),
                              name='reshape_out_' + str(i)
                              )(out)
                # out = Flatten(name='flatten_spacial_policy_' + str(i))(out)
                # normalize the spacial action output layer
                out = Activation(activation='softmax',
                                 name='output_spacial_policy_' + str(i)
                                 )(out)
                output_layers.append(out)
            elif size < 256:
                # this must be a non spacial output action
                non_spacial = Dense(size,
                                    activation='relu',
                                    name='non_spacial_policy_' + str(i))(d1)
                output_layers.append(non_spacial)

        # set and compile model
        model = Model(inputs=sc_i,
                      outputs=output_layers)
        model.compile(optimizer='adam',
                      loss=loss)
        print("new NN generated, file", file_path, "not found")
    return model


def save_neural_network(model):
    save_model(model=model,
               filepath=current_neural_network_file)
