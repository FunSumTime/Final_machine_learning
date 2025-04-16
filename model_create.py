import tensorflow as tf

import keras

def create_model(my_args,input_shape):
    create_functions = {
        "a" : create_model_a,
        "b" : create_model_b,
    }

    if my_args.model_name not in create_functions:
        raise InvalidArgumentException("Invalid model name: {} not in {}".format(my_args.model_name, list(create_functions.keys())))
    
    model = create_functions[my_args.model_name](my_args, input_shape)
    print(model.summary())
    return model


lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=800,
    decay_rate=0.9)

def create_model_a(my_args,input_shape):
    # make network that has two outputs so probalbly binary classification
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3,3), kernel_initializer="he_normal", padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation="relu"))
    model.add(keras.layers.Dropout(.1))

    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation="relu"))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation="relu"))
    model.add(keras.layers.Dropout(.3))

    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation="relu"))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation="relu"))
    model.add(keras.layers.Dropout(.3))


    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(256, kernel_initializer="he_normal"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation(activation="relu"))
    model.add(keras.layers.Dropout(.5))


    model.add(keras.layers.Dense(1, activation="sigmoid"))
    
    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=keras.optimizers.Adam(learning_rate=lr_schedule, amsgrad=True))
    return model
def create_model_b(my_args,input_shape):
    pass 
