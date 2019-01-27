import numpy as np
import tensorflow as tf
import pickle
import cv2

class datasetWriter(object):

    tfr_file = 'cpm_sample_dataset.tfrecords'
    num_of_joints = 7

    image_feature_description = {
        'image': tf.FixedLenFeature([], tf.string),
        'keypoints': tf.FixedLenFeature([num_of_joints * 2], tf.float32),
    }

    def __init__(self, input_size):
        self.data = [] # data[0] = image_array data[1] = keypoints_array
        self.input_size = input_size

        raw_dataset = self._read_tf_records()
        self._create_training_data_from_tf_records(raw_dataset)

        with open('cpm_dataset', 'wb') as fp:
            pickle.dump(self.data, fp)



    def _parse_image_function(self, example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.parse_single_example(example_proto, self.image_feature_description)

    def _read_tf_records(self):
        raw_dataset = tf.data.TFRecordDataset(self.tfr_file)
        parsed_image_dataset = raw_dataset.map(self._parse_image_function)
        return parsed_image_dataset

    def _create_training_data_from_tf_records(self, tf_record_data):
        for image_features in tf_record_data:
            image_raw = image_features['image'].numpy()
            np_image_array = np.frombuffer(image_raw, np.uint8).reshape(self.input_size, self.input_size, 3).astype(np.float32)

            kp = image_features['keypoints'].numpy()
            kp_out = []
            for kp_idx in range(self.num_of_joints):
                kp_out.append([kp[kp_idx * 2], kp[kp_idx * 2 + 1]])

            self.data.append([np_image_array, kp_out])

