'''
Helpers.py contains diverse functions useful for the project. It is sorted the following way:
	-IMPORTS AND VARIABLES
	-IMAGE GENERATING FUNCTION
	-LOSS FUNCTIONS
	-F1-METRIC FUNCTIONS
	-SUBMISSIONS FUNCTIONS

'''


############################################ IMPORTS AND VARIABLES #############################################

import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import os.path
from os import path
from tensorflow.keras import backend as K
from PIL import Image
from tqdm import tqdm
from skimage.transform import rotate, resize
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow
from skimage.transform import resize
from smooth_tiled_predictions import cheap_tiling_prediction
import re
from numpy import load
from numpy import asarray
from numpy import savez_compressed

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#disable tensorflow logs.
tf.get_logger().setLevel('INFO')

#threshold to assign label road or other to a pixel.
foreground_threshold = 0.49 #percentage of pixels > 1 required to assign a foreground label to a patch


IMG_SIZE = 400
IMG_CHANNELS = 3
IMG_PATCH_SIZE = 16
SEED = 66478

train_labels_path = "../data/labels/"
train_data_path = "../data/images/"

data_dir = "../data"



############################################ IMAGE GENERATING FUNCTION #############################################


def generate_images(save_to, imgs_number):
	'''
	Image generating function, it will save generated images in a given folder.
	:param save_to: folder name where to save the images
	:param n: number of images generated per source image ( n= 10 will result in 100*10=1000 images)
	'''

	source_images = os.listdir(train_data_path)
	source_groundtruth = os.listdir(train_labels_path)
	BATCH_SIZE = 32

	if not path.exists('../data/{}'.format(save_to)):
		os.mkdir('../data/{}'.format(save_to))
	if not path.exists('../data/{}/images'.format(save_to)):
		os.mkdir('../data/{}/images'.format(save_to))
	else:
		if (len(os.listdir('../data/{}/images'.format(save_to)))==100*imgs_number):
			print("Existing images found!")
			return 0

	if not path.exists('../data/{}/labels'.format(save_to)):
		os.mkdir('../data/{}/labels'.format(save_to))
	new_img_folder = '../data/{}/images'.format(save_to)
	new_gt_folder = '../data/{}/labels'.format(save_to)

	for n, id_ in tqdm(enumerate(source_images), total=len(source_images)):
		img = imread(train_data_path + id_ )[:,:,:IMG_CHANNELS]
		img = resize(img, (IMG_SIZE, IMG_SIZE), mode='constant', preserve_range=True)
		img = np.expand_dims(img, axis=0)
		mask = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.bool)
		mask_ = imread(train_labels_path + id_)
		mask_ = np.expand_dims(resize(mask_, (IMG_SIZE, IMG_SIZE), mode='constant',
		                                    preserve_range=True), axis=-1)
		mask = np.maximum(mask, mask_)
		mask = np.expand_dims(mask, axis=0)

      # This is the image data genenator where you can tweak the parameters
		datagen = ImageDataGenerator(
		rotation_range=360,
		width_shift_range=0.2,
		height_shift_range=0.2,
		zoom_range=0.6,
		fill_mode="reflect",
		horizontal_flip=True,
		vertical_flip=True,
		)
		imageGenerated = datagen.flow(
		img,
		y=mask,
		batch_size=BATCH_SIZE,
		shuffle=True,
		seed=SEED,
		save_to_dir=new_img_folder,
		save_prefix=str(n) ,
		save_format="png",
		)
		groundTruthGenerated = datagen.flow(
		mask,
		y=mask,
		batch_size=BATCH_SIZE,
		shuffle=True,
		seed=SEED,
		save_to_dir=new_gt_folder,
		save_prefix= str(n),
		save_format="png",
		)
		totalgenerated=0

		for image in imageGenerated:
			totalgenerated+=1
			if (totalgenerated >= imgs_number):
				totalgenerated=0
				break 
		
		for image in groundTruthGenerated:   
			totalgenerated+=1
			if (totalgenerated >= imgs_number):
				totalgenerated=0
				break 


def normalize(img):
	'''
	Standardize the image ( Z-score)
	:param img: image:
	:return out: standardized image
	'''
	mean = np.mean(img, axis=(1,2), keepdims=True)
	std = np.std(img, axis=(1,2), keepdims=True)
	out = (img - mean) / std
	return out



############################################ LOSS FUNCTIONS ###################################################
def jaccard_loss(y_true, y_pred,  smooth=1):
	'''
	jaccard loss function
	:param y_true: true labels 
	:param y_pred: predicted labels
	:return: computed loss
	'''
	y_true_int = tf.where(y_true==True, 1., 0.)
	y_true_f = tf.keras.backend.flatten(y_true_int)
	y_pred_f = tf.keras.backend.flatten(y_pred)

	intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
	jaccard = (intersection +  smooth) / (tf.keras.backend.sum(y_true_f)   +  tf.keras.backend.sum(y_pred_f) - intersection + smooth)

	return 1-jaccard

def dice_loss(y_true, y_pred, smooth = 1):
	'''
	dice loss function
	:param y_true: true labels 
	:param y_pred: predicted labels
	:return: computed loss
	'''
	y_true_int = tf.where(y_true==True, 1., 0.)
	y_true_f = tf.keras.backend.flatten(y_true_int)
	y_pred_f = tf.keras.backend.flatten(y_pred)

	intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
	dice = (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f)   +  tf.keras.backend.sum(y_pred_f) + smooth)

	return 1-dice

def jaccard_and_binary_crossentropy(y_true, y_pred):
	'''
	jaccard and binary crossentropy loss function
	:param y_true: true labels 
	:param y_pred: predicted labels
	:return: computed loss
	'''
	return tf.keras.losses.binary_crossentropy(y_true, y_pred) + jaccard_loss(y_true, y_pred)

def dice_and_binary_crossentropy(y_true, y_pred):
	'''
	dice and binary crossentropy loss function
	:param y_true: true labels 
	:param y_pred: predicted labels
	:return: computed loss
	'''
	return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def load_images(folder_name = "../data" , norm = False):
	'''
	load images loads the images and their corresponding groundtruth from a folder
	and returns them as numpy arrays
	:param folder_name: folder containing both folders of images and groundtruths
	:return x_train,y_train: numpy arrays of the images and their labels
	'''
	train_data_path = "../data/{}/images".format(folder_name)
	train_labels_path = "../data/{}/labels".format(folder_name)

	IMAGE_OPEN = os.listdir(train_data_path)

	X = np.zeros((len(IMAGE_OPEN), IMG_SIZE, IMG_SIZE, IMG_CHANNELS), dtype=np.uint8)
	Y = np.zeros((len(IMAGE_OPEN), IMG_SIZE, IMG_SIZE, 1), dtype=np.bool)
	train_data_path
	for n, id_ in tqdm(enumerate(IMAGE_OPEN), total=len(IMAGE_OPEN)):
		try:
			path = data_dir
			img = imread(train_data_path + id_ )[:,:,:IMG_CHANNELS]
			img = resize(img, (IMG_SIZE, IMG_SIZE), mode='constant', preserve_range=True)
			X[n] = img
			mask = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.bool)
			mask_ = imread(train_labels_path + id_)
			mask_ = np.expand_dims(resize(mask_, (IMG_SIZE, IMG_SIZE), mode='constant',
                            preserve_range=True), axis=-1)
			mask = np.maximum(mask, mask_)
			Y[n] = mask
		except:
			pass
		x_train=X
		y_train=Y

	if norm:
		print("Nomalizing images")
		for i in tqdm(range(len(x_train))):
			x_train[i] = normalize(x_train[i])

	return x_train, y_train

############################################ F1-METRIC FUNCTIONS ###################################################


def recall_m(y_true, y_pred):
    '''
    computes the recall
    source: https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
    :param y_true: real label
    :param y_pred: prediction
    :return recall: the computed recall
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    '''
    computes the precision
    source: https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
    :param y_true: real label
    :param y_pred: prediction
    :return recall: the computed precision
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    '''
    computes the F1_score
    source: https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
    :param y_true: real label
    :param y_pred: prediction
    :return : the computed f1_score
    '''
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def load_image(infilename):
    data = mpimg.imread(infilename)
    return data





    


############################################ SUBMISSIONS FUNCTIONS ###############################################
def patch_to_label(patch):
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename, index):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(index, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        index = 1
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn, index))
            index+=1



def create_submission(model, subname, norm):
	'''
	Create a submission file to submit to Aicrowd website
	:param model: the trained model
	:param subname: the submission file name
	'''
	test_dir_images = "../data/test/test_set_images/"
	test_dir_groundtruth = "../data/test/test_set_prediction/"
	submission_filename = "../submissions/" + subname
	image_filenames = []
	for i in tqdm(range(1, 51)):
	    x = imread(test_dir_images + "test_{}.png".format(i))[:,:,:IMG_CHANNELS]
	    if norm:
	    	x = normalize(x)
	    predictions_smooth = cheap_tiling_prediction(x, IMG_SIZE, 1, pred_func=(
	      lambda img_batch_subdiv: model.predict(np.expand_dims(img_batch_subdiv, axis=0)[:,0,:,:])
	      )
	    )
	    image_filename = test_dir_groundtruth + "pred_{}.png".format(i)
	    mpimg.imsave( image_filename, np.squeeze(predictions_smooth))

	    image_filenames.append(image_filename)
	masks_to_submission(submission_filename, *image_filenames)


