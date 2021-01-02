import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np

from tqdm import tqdm

from itertools import chain

from skimage.io import imread, imshow
from skimage.transform import resize

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

from numpy import load
from numpy import asarray
from numpy import savez_compressed

import tensorflow as tf
from tensorflow import keras

import re

from unet import unet
from helpers import *

from PIL import Image



from sklearn.model_selection import train_test_split
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as keras
import argparse

import warnings
import logging
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings("ignore")


directoryOfProject = '../'
data_dir = directoryOfProject + 'data/'

IMG_CHANNELS = 3
IMG_SIZE = 400
DIMS = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
SEED = 66478


class bcolors:
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def run( model_filename,subname, train, EPOCH_NUMBER, STEP_PER_EPOCH, loss, IMAGES_GEN_NUMBER,NEW_IMAGES_FOLDER, BATCH_SIZE, norm):
    '''
    main function, can either launch the training or load a defined model, 
    then will compute a submission file
    
    :param model_filename: name of the model (to save as or to load)
    :param subname: name of the submission file
    :param train: boolean, if true launch new training
    :param EPOCH_NUMBER: number of epochs
    :param STEP_PER_EPOCH: number of step per epoch
    :param loss: loss to use during training
    :BATCH_SIZE: batch size
    :param norm: boolean, if true images are normalized
    :return: 1
    '''
    if (not train):
        list_models = os.listdir("../models/")
        list_models = re.findall('([^\s]+).index', ' '.join(list_models))
        if model_filename not in list_models:
            print(bcolors.FAIL + "[ERROR]" + bcolors.ENDC+"  Model not found")
            print("Please choose a model in the following ones:", list_models)
            print("or train a new model by appending flag -train")
            exit()
    if (loss == 'binary_ce'):
        loss_loaded = tf.keras.losses.binary_crossentropy
    elif (loss == 'dice_bce'):
        loss_loaded = dice_and_binary_crossentropy
    elif (loss == 'jaccard_bce'):
        loss_loaded = jaccard_and_binary_crossentropy
    else:
        print(bcolors.FAIL + "[ERROR]" + bcolors.ENDC+"  Loss not found")
        print("possibles loss are: binary_ce, dice_bce, jaccard_bce")
        exit()


    if train is False:
        print("loading model")
        model = unet(DIMS)
        model.load_weights("../models/{}".format(model_filename))
    else:
        print("generating dataset in {} folder".format(NEW_IMAGES_FOLDER))
        generate_images(NEW_IMAGES_FOLDER, IMAGES_GEN_NUMBER)
        model = train_(model_filename, EPOCH_NUMBER ,STEP_PER_EPOCH,loss_loaded,BATCH_SIZE, NEW_IMAGES_FOLDER)
    print("Creating Submission file")
    create_submission(model, subname, norm)
    print(bcolors.OKGREEN + "[Success]" + bcolors.ENDC + "  Submission file successfully created")

    return 1
        
    
    
def train_(model_filename, EPOCH_NUMBER ,STEP_PER_EPOCH, loss_loaded, BATCH_SIZE, NEW_IMAGES_FOLDER, norm):
    '''
    training function. 
    
    :param model_filename: name of the model (to save as or to load)
    :param EPOCH_NUMBER: number of epochs
    :param STEP_PER_EPOCH: number of step per epoch
    :param loss: loss to use during training
    :BATCH_SIZE: batch size
    :param norm: boolean, if true normalizes the data
    :return model: the model weights after training
    '''
        

    print("Loading images")
    x_train, y_train = load_images(NEW_IMAGES_FOLDER, norm)

    print(f"Beginning training using the following parameters:\n    Model filename: {model_filename},\n    Number of epoch: {EPOCH_NUMBER},\n    Steps per epochs: {STEP_PER_EPOCH},\n    Batch size: {BATCH_SIZE}")

    model = unet(DIMS)
    
    cp = ModelCheckpoint(model_filename, verbose=1, monitor='val_loss', save_best_only=True)
    lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, verbose=1, mode='min', min_lr= 1e-9)
    es = EarlyStopping(monitor = 'val_loss', patience = 25, mode = 'min')

    model.compile(optimizer='adam', loss=loss_loaded, metrics=['accuracy', f1_m])

    checkpoint_path = "train_checkpoints/{}.ckpt".format(model_filename)
    checkpoint_dir = os.path.dirname(checkpoint_path)


    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                     save_weights_only=True,
                                                     verbose=1)

    callbacks = [
      tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
      tf.keras.callbacks.TensorBoard(log_dir='./logs'),
      cp_callback 
    ]
    results = model.fit(x_train, y_train, validation_split=0.2, batch_size=BATCH_SIZE, epochs=EPOCH_NUMBER,shuffle=True,
                    callbacks=callbacks)

    
    return model

def parse_args():
    """
    Parse command line flags.
    :return results: Namespace of the arguments to pass to the main run function.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', default='bestModel.ckpt', dest='model_filename', help='Name of the model')
    parser.add_argument('-train', action='store_true', default=False, dest='train', help='Train model from scratch')
    parser.add_argument('-norm', action='store_true', default=False, dest='norm', help='use standardization of images')

    parser.add_argument('-epochs', type=int, dest='EPOCH_NUMBER',default=120, help='Number of epoch')
    parser.add_argument('-steps', type=int, dest='STEPS_PER_EPOCH',default=100, help='Number of steps per epoch')
    parser.add_argument('-loss', type=str, dest='loss', default='binary_ce' ,help='Loss used in training')
    parser.add_argument('-batch',type=int, default=2, dest='BATCH_SIZE', help='Batch size')
    parser.add_argument('-images',type=int, default=80, dest='IMAGES_GEN_NUMBER', help='Number of new images generated per source image')
    parser.add_argument('-imFolder', type=str, dest='NEW_IMAGES_FOLDER', default='training_extra' ,help='name of new images folder')
    parser.add_argument('-sub', type=str, dest='subname', default='submission' ,help='submission file name')



    results = parser.parse_args()

    return results

if __name__ == '__main__':
    args = parse_args()
    run(args.model_filename, args.subname, args.train, args.EPOCH_NUMBER, args.STEPS_PER_EPOCH,args.loss,args.IMAGES_GEN_NUMBER, args.NEW_IMAGES_FOLDER, args.BATCH_SIZE, args.norm)















