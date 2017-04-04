## create script that download datasets and transform into tf-record
## Assume the datasets is downloaded into following folders
## SythTexts datasets(41G)
## data/sythtext/*

import numpy as np 
import scipy.io as sio
import os, os.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import tensorflow as tf
import re
from datasets.dataset_utils import int64_feature, float_feature, bytes_feature ,ImageCoder, norm

from PIL import Image

data_path = 'data/train'
os.chdir(data_path)
fish_folder = ['ALB', 'BET', 'DOL', 'LAG','OTHER','SHARK','YFT']
labels = [i+1 for i in range(len(fish_folder))]


## SythText datasets is too big to store in a record. 
## So Transform tfrecord according to dir name


def _convert_to_example(image_data, shape, bbox, label):
	nbbox = np.array(bbox)
	ymin = list(nbbox[:, 0])
	xmin = list(nbbox[:, 1])
	ymax = list(nbbox[:, 2])
	xmax = list(nbbox[:, 3])

	#print 'shape: {}, height:{}, width:{}'.format(shape,shape[0],shape[1])
	example = tf.train.Example(features=tf.train.Features(feature={
			'image/shape': int64_feature(list(shape)),
			'image/object/bbox/ymin': float_feature(ymin),
			'image/object/bbox/xmin': float_feature(xmin),
			'image/object/bbox/ymax': float_feature(ymax),
			'image/object/bbox/xmax': float_feature(xmax),
			'image/object/bbox/label': int64_feature(label),
			'image/format': bytes_feature('jpeg'),
			'image/encoded': bytes_feature(image_data),
			}))
	return example
	

def run():
	coder = ImageCoder()
	tfrecord_writer = tf.python_io.TFRecordWriter('fish.tfrecord')
	for i, name in enumerate(fish_folder):
		mat_file = name + '_label.mat'
		fish_mat = sio.loadmat(mat_file)
		fish_list = fish_mat[name]
		num = fish_list.shape[1]
		for k in xrange(num):

			imname = str(fish_list[0, k][0][0])
			image_data = tf.gfile.GFile(imname, 'r').read()
			print imname
			image = coder.decode_jpeg(image_data)
			shape = image.shape

			ori_box = fish_list[0, k][1]
			ymin = ori_box[:, 1]
			xmin = ori_box[:, 0]
			w = ori_box[:,2]
			h = ori_box[:,3]
			ymax = ymin + h
			xmax = xmin + w
			xmin = np.maximum(xmin*1.0/shape[1], 0.0)
			ymin = np.maximum(ymin*1.0/shape[0], 0.0)
			xmax = np.minimum(xmax*1.0/shape[1], 1.0)
			ymax = np.minimum(ymax*1.0/shape[0], 1.0)
			bbox = np.vstack([ymin, xmin, ymax, xmax]).T

			numofbox = bbox.shape[0]
			label = [i+1 for l in range(numofbox)]
			print label
			example = _convert_to_example(image_data, shape, bbox, label)
			tfrecord_writer.write(example.SerializeToString()) 

	print 'Transform to tfrecord finished'

if __name__ == '__main__':
	run()





