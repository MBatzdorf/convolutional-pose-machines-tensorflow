from .augmentors import ScaleAug, RotateAug, HueAug, FlipAug, SaturationAug, \
    joints_to_point8, point8_to_joints, AugImgMetadata
import cv2
import numpy as np
from random import uniform

class PenAugmentor(object):
    """
    Holder for data required for augmentation - subset of metadata
    """

    augmentation_propability = 0.6

    def __init__(self, aug_config):
        rotation_limit = aug_config['rotate_limit']

        scale_min =aug_config['scale_limit'][0]
        scale_max = aug_config['scale_limit'][1]

        saturation_min = aug_config['sat_shift_limit'][0]
        saturation_max = aug_config['sat_shift_limit'][1]

        hue_min = aug_config['hue_shift_limit'][0]
        hue_max = aug_config['hue_shift_limit'][1]

        self.rotate_aug = RotateAug(rotate_max_deg=rotation_limit,
                  interp=cv2.INTER_CUBIC,
                  border=cv2.BORDER_CONSTANT,
                  border_value=(128, 128, 128))

        self.scale_aug = ScaleAug(scale_min=scale_min,
                 scale_max=scale_max,
                 target_dist=1,
                 interp=cv2.INTER_CUBIC)

        self.flip_aug = FlipAug(num_parts=2, prob=0.5)

        self.saturation_aug = SaturationAug(saturation_min, saturation_max)

        self.hue_aug = HueAug(hue_min, hue_max)

    def augment_image_and_keypoints(self, image, keypoints):
        kp = keypoints
        img = image

        if uniform(0, 1) <= self.augmentation_propability:
            img = self.change_hue(img)

        if uniform(0, 1) <= self.augmentation_propability:
            img = self.change_saturation(img)

        img, kp = self.flip_image(img, kp)

        if uniform(0, 1) <= self.augmentation_propability:
            img, kp = self.rotate_image(img, kp)

        return img, kp

    def change_hue(self, image):
        im, params = self.hue_aug.augment_return_params(
            AugImgMetadata(img=image))
        return im

    def change_saturation(self, image):
        im, params = self.saturation_aug.augment_return_params(
            AugImgMetadata(img=image))
        return im

    def flip_image(self, image, keypoints):
        im, params = self.flip_aug.augment_return_params(
            AugImgMetadata(img=image))
        aug_joints = self.flip_aug.augment_coords(np.asarray(keypoints), params)
        return im, aug_joints

    def rotate_image(self, image, keypoints):
        im, params = self.rotate_aug.augment_return_params(
            AugImgMetadata(img=image))
        aug_joints = self.rotate_aug.augment_coords(np.asarray(keypoints), params)
        return im, aug_joints

    def scale_image(self, image, keypoints):
        im, params = self.scale_aug.augment_return_params(
            AugImgMetadata(img=image))
        aug_joints = self.scale_aug.augment_coords(np.asarray(keypoints), params)
        return im, aug_joints