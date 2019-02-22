import numpy as np
import pickle
import cv2
from .pen_augmentor import PenAugmentor

class dataProvider(object):

    num_of_joints = 7
    input_dataset_file = 'cpm_dataset_cropped'
    input_validation_dataset_file = 'cpm_dataset_cropped_validation'


    def __init__(self, batch_size, aug_config, for_validation):
        self.data = [] # data[0] = image_array data[1] = keypoints_array
        self.batch_size = batch_size
        self.aug_config = aug_config
        self.current_index = 0
        self.pen_augmentor = PenAugmentor(aug_config)

        if for_validation:
            with open(self.input_validation_dataset_file, 'rb') as fp:
                self.data = pickle.load(fp)
        else:
            with open(self.input_dataset_file, 'rb') as fp:
                self.data = pickle.load(fp)


    def next(self):
        imgs = []
        kps = []
        for i in range(self.batch_size):
            if self.current_index >= len(self.data):
                self.current_index = 0
            current_image = self.data[self.current_index][0]
            current_keypoints = self.data[self.current_index][1]

            new_img, new_kp = self.pen_augmentor.augment_image_and_keypoints(current_image, current_keypoints)

            imgs.append(new_img)
            kps.append(new_kp)

            self.current_index = self.current_index + 1

        return np.asarray(imgs).astype(np.float32), np.asarray(kps).astype(np.float32)


'''
# TODO: Remove - just for testing
augmentation_config = {'hue_shift_limit': (-5, 5),
                           'sat_shift_limit': (0, 127),
                           'val_shift_limit': (-15, 15),
                           'translation_limit': (-0.15, 0.15),
                           'scale_limit': (0.5, 1.1),
                           'rotate_limit': 90}

dp = dataProvider(5, augmentation_config)
dp.next()
'''
