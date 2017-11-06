from os import listdir
import numpy as np
from cv2 import imread, imshow, waitKey

def parse(path):
	'''
	read all images in the folder(path)
	input:
		- path: the directory of the images
	output:
		- imgs: list of images
	'''
	imgs = []
	for imgname in listdir(path):
		img = imread(str(path+'/'+imgname), 0)
		imgs.append(img)
	return imgs

def read_data():
	'''
	read train and test data and masks.
	'''
	x_train = parse('../data/2d_images')
	y_train = parse('../data/2d_masks')
	return np.array(x_train), np.array(y_train)

'''
def tester():
	imgs = parse('./data/2d_images')
	part = imgs[0:3]
	part = np.array(part,np.uint8)
	print part
	print part.shape
	print len(imgs)
'''
#tester()