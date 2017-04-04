## an initial version
## Transform the tfrecord to slim data provider format

import numpy 
import tensorflow as tf
import os
slim = tf.contrib.slim




ITEMS_TO_DESCRIPTIONS = {
    'image': 'slim.tfexample_decoder.Image',
    'shape': 'shape',
    'height': 'height',
    'width': 'width',
    'object/bbox': 'box',
    'object/label': 'label'
}
SPLITS_TO_SIZES = {
    'train': 20000,
}
NUM_CLASSES = 7



def get_datasets(data_dir,file_pattern = '*.tfrecord'):
    file_patterns = os.path.join(data_dir, file_pattern)
    print 'file_path: {}'.format(file_patterns)
    reader = tf.TFRecordReader
    keys_to_features = {
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/format': tf.FixedLenFeature([], tf.string, default_value='jpeg'),
        'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
      }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None


    return slim.dataset.Dataset(
        data_sources=file_patterns,
        reader=reader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZES['train'],
        items_to_descriptions=ITEMS_TO_DESCRIPTIONS,
        num_classes=NUM_CLASSES,
        labels_to_names=labels_to_names)