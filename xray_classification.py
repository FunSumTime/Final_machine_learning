import sys
import argparse
import logging


import tensorflow as tf
import keras
import joblib

import open_data
import model_create



def do_fit(my_args):

    X, y = open_data.load_dataset(my_args.data_set)
    model = model_create.create_model(my_args,X.shape[1:])
    # earyl stoping
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # history
    history = model.fit(X, y, epochs=5, verbose=1, callbacks=[early_stopping], validation_split=0.2, shuffle=True, batch_size=4)

    model_file = my_args.model_file
    joblib.dump(model, model_file)
    joblib.dump(history.history, "{}.history".format(model_file))

    # dump history as well
    return

import model_history


def parse_args(argv):
    parser = argparse.ArgumentParser(prog=argv[0], description='Image Classification with CNN')
    parser.add_argument('action', default='cnn-fit',
                        choices=[ "cnn-fit", "score", "learning-curve", "cnn-refit" ], 
                        nargs='?', help="desired action")

    parser.add_argument('--data-set',  '-d', default=2,     type=int,   help="which dataset to grab 2 is training")
    parser.add_argument('--model-file',    '-m', default="model.joblib",    type=str,   help="name of file for the model (default is constructed from train file name when fitting)")
    parser.add_argument('--model-name',    '-M', default="a",    type=str,   help="name of model create function")

    my_args = parser.parse_args(argv[1:])

    #
    # Do any special fixes/checks here
    #
    
    return my_args



def main(argv):
    my_args = parse_args(argv)
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARN)

    if my_args.action == 'cnn-fit':
        do_fit(my_args)
    # elif my_args.action == 'cnn-refit':
    #     do_cnn_refit(my_args)
    elif my_args.action == 'score':
        model_evaluation.show_score(my_args)
    elif my_args.action == 'learning-curve':
        model_history.plot_history(my_args)
    else:
        raise Exception("Action: {} is not known.".format(my_args.action))

    return

if __name__ == "__main__":
    main(sys.argv)

