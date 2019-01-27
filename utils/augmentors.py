import numpy as np
import cv2

from tensorpack.dataflow.imgaug.base import ImageAugmentor
from tensorpack.dataflow.imgaug.transform import ResizeTransform
from tensorpack.dataflow.imgaug.geometry import WarpAffineTransform,TransformAugmentorBase


class AugImgMetadata:
    """
    Holder for data required for augmentation - subset of metadata
    """
    __slots__ = ["img"]

    def __init__(self, img):
        self.img = img

    def update_img(self, new_img):
        return AugImgMetadata(new_img)


def joints_to_point8(joints, num_p=8):
    """
    Converts joints structure to Nx2 nparray (format expected by tensorpack augmentors)
    Nx2 = floating point nparray where each row is (x, y)

    :param joints:
    :param num_p:
    :return: Nx2 nparray
    """
    segment = np.zeros((num_p * len(joints), 2), dtype=float)

    for idx_all, j_list in enumerate(joints):
        for idx, k in enumerate(j_list):
            if k:
                segment[idx_all * num_p + idx, 0] = k[0]
                segment[idx_all * num_p + idx, 1] = k[1]
            else:
                segment[idx_all * num_p + idx, 0] = -1000000
                segment[idx_all * num_p + idx, 1] = -1000000

    return segment


def point8_to_joints(points, num_p=8):
    """
    Converts Nx2 nparray to the list of joints

    :param points:
    :param num_p:
    :return: list of joints [[(x1,y1), (x2,y2), ...], []]
    """
    l = points.shape[0] // num_p

    all = []
    for i in range(l):
        skel = []
        for j in range(num_p):
            idx = i * num_p + j
            x = points[idx, 0]
            y = points[idx, 1]

            if x <= 0 or y <= 0 or x > 2000 or y > 2000:
                skel.append(None)
            else:
                skel.append((x, y))

        all.append(skel)
    return all


class FlipAug(ImageAugmentor):
    """
    Flips images and coordinates
    """
    def __init__(self, num_parts, prob=0.5):
        super(FlipAug, self).__init__()
        self._init(locals())

    def _get_augment_params(self, meta):
        img = meta.img

        _, w = img.shape[:2]

        do = self._rand_range() < self.prob
        return (do, w)

    def _augment(self, meta, param):
        img = meta.img

        do, _ = param

        if do:
            new_img = cv2.flip(img, 1)
            if img.ndim == 3 and new_img.ndim == 2:
                new_img = new_img[:, :, np.newaxis]

            result = new_img
        else:
            result = img

        return result

    def _augment_coords(self, coords, param):
        do, w = param
        if do:
            coords[:, 0] = w - coords[:, 0]

        return coords

    def recover_left_right(self, coords, param):
        """
        Recovers a few joints. After flip operation coordinates of some parts like
        left hand would land on the right side of a person so it is
        important to recover such positions.

        :param coords:
        :param param:
        :return:
        """
        do, _ = param
        if do:
            right = [3, 6]
            left = [2, 5]

            for l_idx, r_idx in zip(left, right):
                idxs = range(0, coords.shape[0], self.num_parts)
                for idx in idxs:
                    tmp = coords[l_idx + idx, [0, 1]]
                    coords[l_idx + idx, [0, 1]] = coords[r_idx + idx, [0, 1]]
                    coords[r_idx + idx, [0, 1]] = tmp

        return coords


class ScaleAug(TransformAugmentorBase):
    def __init__(self, scale_min, scale_max, target_dist = 1.0, interp=cv2.INTER_CUBIC):
        super(ScaleAug, self).__init__()
        self._init(locals())

    def _get_augment_params(self, meta):
        img = meta.img

        h, w = img.shape[:2]

        scale_multiplier = self._rand_range(self.scale_min, self.scale_max)

        scale_abs = self.target_dist

        scale = scale_abs * scale_multiplier

        new_h, new_w = int(scale * h + 0.5), int(scale * w + 0.5)

        return ResizeTransform(
            h, w, new_h, new_w, self.interp)

    def _augment(self, meta, params):
        new_img = params.apply_image(meta.img)

        return new_img


class RotateAug(TransformAugmentorBase):
    """
    Rotates images and coordinates
    """
    def __init__(self, scale=None, translate_frac=None, rotate_max_deg=0.0, shear=0.0,
                 interp=cv2.INTER_LINEAR, border=cv2.BORDER_REPLICATE, border_value=0):

        super(RotateAug, self).__init__()
        self._init(locals())
        self.scale = 1

    def _get_augment_params(self, meta):
        img = meta.img

        # grab the rotation matrix
        (h, w) = img.shape[:2]
        (center_x, center_y) = (w // 2, h // 2)
        deg = self._rand_range(-self.rotate_max_deg, self.rotate_max_deg)
        R = cv2.getRotationMatrix2D((center_x, center_y), deg, 1.0)

        # determine bounding box
        (h, w) = img.shape[:2]
        cos = np.abs(R[0, 0])
        sin = np.abs(R[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        R[0, 2] += (new_w / 2) - center_x
        R[1, 2] += (new_h / 2) - center_y

        return WarpAffineTransform(R, (new_w, new_h),
                            self.interp, self.border, self.border_value)

    def _augment(self, meta, params):
        new_img = params.apply_image(meta.img)
        self.scale = meta.img.shape[0] / new_img.shape[0]
        return cv2.resize(new_img, meta.img.shape[:2])
        return new_img

    def _augment_coords(self, coords, param):
        unscaled = super(RotateAug, self)._augment_coords(coords, param)
        scaled = unscaled * self.scale
        return scaled


def _modifyHSVChanel(image, chan_idx, val):
    img = image
    hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
    #hsv[..., chan_idx] = (hsv[..., chan_idx] + val)
    hsv[..., chan_idx] = val
    new_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    new_image = np.clip(new_image, 0, 255)
    return new_image

class SaturationAug(ImageAugmentor):

    def __init__(self, sat_min = 0, sat_max = 0):

        super(SaturationAug, self).__init__()
        self._init(locals())

    def _get_augment_params(self, meta):
        return self._rand_range(self.sat_min, self.sat_max)

    def _augment(self, meta, params):
        return _modifyHSVChanel(meta.img, 1, params)


class HueAug(ImageAugmentor):

    def __init__(self, hue_min = 0, hue_max = 0):

        super(HueAug, self).__init__()
        self._init(locals())

    def _get_augment_params(self, meta):
        return self._rand_range(self.hue_min, self.hue_max)

    def _augment(self, meta, params):
        return _modifyHSVChanel(meta.img, 0, params)