from tensorflow.keras.initializers import Constant, TruncatedNormal, HeNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, BatchNormalization, concatenate, Conv2D, Dropout, Input, PReLU

##################### STAN's MODEL #####################

stride = (1, 1)

shapes_layers = [[32, (1,1)],
                 [32, (3,3)],
                 [16, (5,5)],
                 [8, (7,7)],
                 [4, (9,9)],
                 [2, (11,11)],
                 [1, (13,13)]]

shapes_layers_final = [[8, (8,8)],
                       [4, (16,16)],
                       [2, (32,32)],
                       [1, (64,64)]]

# init_kernel = HeNormal(seed=420)
init_kernel = TruncatedNormal(mean=0.0, stddev=0.03, seed=420)
init_bias = Constant(value=0.01)

def block(data, filters, shapes, active="None", drop=False):
    # INCEPTION LAYER
    layers = []
    for f, s in shapes:
        out_layer = Conv2D(filters = f, kernel_size = s, strides = stride,
                           kernel_initializer = init_kernel, bias_initializer = init_bias,
                           padding = "same", activation = None)(data)

        # ACTIVATION FUNCTION
        if active == "relu": out_layer = Activation("relu")(out_layer)
        elif active == "prelu": out_layer = Activation(PReLU())(out_layer)

        out_layer = BatchNormalization(axis = -1)(out_layer)

        layers.append(out_layer)

    # CONCATENATE INCEPTION FILTERS
    layers = concatenate(layers, axis=-1)
    
    # CONSEQUENT CONV LAYER
    out_layer = Conv2D(filters = filters, kernel_size = (1,1), strides = stride, 
                       kernel_initializer = init_kernel, bias_initializer = init_bias, 
                       padding = "same", activation = None)(layers)

    # ACTIVATION FUNCTION
    if filters > 1:
        if active == "relu": out_layer = Activation("relu")(out_layer)
        elif active == "prelu": out_layer = Activation(PReLU())(out_layer)

        out_layer = BatchNormalization(axis = -1)(out_layer)

        if drop: out_layer = Dropout(drop)(out_layer)
    
    else: out_layer = Activation("sigmoid")(out_layer)

    return out_layer

def Stannet(shape_image, active="None", drop=False):
    input_data = Input(shape=(shape_image))
    data = BatchNormalization(axis = -1)(input_data)
    data = block(data, 32, shapes_layers, active, drop)
    data = block(data, 64, shapes_layers, active, drop)
    data = block(data, 64, shapes_layers, active, drop)
    data = block(data, 32, shapes_layers, active, drop)
    data = block(data, 1, shapes_layers_final, active)
    output = Model([input_data], data)
    return output
