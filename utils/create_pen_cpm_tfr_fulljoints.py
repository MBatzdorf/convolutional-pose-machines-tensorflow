import cv2
from pycocotools.coco import COCO
import cpm_utils
import numpy as np
import math
import tensorflow as tf
tf.enable_eager_execution()
from datasetWriter import datasetWriter as dw

import time
import os


tfr_file = 'cpm_sample_dataset.tfrecords'

SHOW_INFO = False
box_size = 368 # according to paper
num_of_joints = 7
gaussian_radius = 2

curr_dir = os.path.dirname(__file__)
annot_path = os.path.abspath(os.path.join(curr_dir, '../dataset/annotations/annotations.json'))
img_dir = os.path.abspath(os.path.join(curr_dir, '../dataset/train2017/'))

annotations = COCO(annot_path)
image_ids = annotations.imgs.keys()

print(curr_dir)
print(annot_path)
print(img_dir)
print(annotations.imgs.keys())


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float64_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# Create writer
tfr_writer = tf.python_io.TFRecordWriter(tfr_file)

img_count = 0
t1 = time.time()
# Loop each dir
for i, img_id in enumerate(image_ids):

    img_meta = annotations.imgs[img_id]
    cur_img_path = img_dir + "/" + img_meta['file_name']

    if not os.path.isfile(cur_img_path):
        print("Not a file")
        continue

    # Check if it is a valid img file
    if not cur_img_path.endswith(('jpg', 'png')):
        print("not an image")
        continue

    cur_img = cv2.imread(cur_img_path)

    annotation_ids = annotations.getAnnIds(imgIds=img_id)
    anns = annotations.loadAnns(annotation_ids)

    for annot_elem in anns:

        if annot_elem["num_keypoints"] is not num_of_joints or annot_elem["area"] < 32 * 32:
            print("Not enough Keypoints")
            continue

        # Read in bbox and joints coords
        cur_pen_bbox = annot_elem["bbox"]

        cur_pen_bbox[2] = cur_pen_bbox[0] + cur_pen_bbox[2]
        cur_pen_bbox[3] = cur_pen_bbox[1] + cur_pen_bbox[3]

        if cur_pen_bbox[0] < 0: cur_pen_bbox[0] = 0
        if cur_pen_bbox[1] < 0: cur_pen_bbox[1] = 0
        if cur_pen_bbox[2] > cur_img.shape[1]: cur_pen_bbox[2] = cur_img.shape[1]
        if cur_pen_bbox[3] > cur_img.shape[0]: cur_pen_bbox[3] = cur_img.shape[0]

        keypoints = annot_elem["keypoints"]
        cur_pen_joints_x = []
        cur_pen_joints_y = []

        for kp_idx in range(annot_elem["num_keypoints"]):
            cur_pen_joints_x.append(float(keypoints[kp_idx * 3]))
            cur_pen_joints_y.append(float(keypoints[kp_idx * 3 + 1]))

        '''
        # Crop image and adjust joint coords
        cur_img = cur_img[int(float(cur_pen_bbox[1])):int(float(cur_pen_bbox[3])),
                  int(float(cur_pen_bbox[0])):int(float(cur_pen_bbox[2])),
                  :]

        cur_pen_joints_x = [x - cur_pen_bbox[0] for x in cur_pen_joints_x]
        cur_pen_joints_y = [x - cur_pen_bbox[1] for x in cur_pen_joints_y]

        '''
        if float(cur_pen_bbox[2] - cur_pen_bbox[0]) > float(cur_pen_bbox[3] - cur_pen_bbox[1]):
            isHorizontal = True
        else:
            isHorizontal = False

        if isHorizontal:
            longside = cur_pen_bbox[2] - cur_pen_bbox[0]
            shortside = cur_pen_bbox[3] - cur_pen_bbox[1]
            difference = longside - shortside
            newylow = cur_pen_bbox[1] - difference / 2
            newyhigh = cur_pen_bbox[3] + difference / 2

            if newylow < 0:
                newylow = 0
            if newyhigh > cur_img.shape[1]:
                newyhigh = box_size

            cur_img = cur_img[int(float(newylow)):int(float(newyhigh)), int(float(cur_pen_bbox[0])):int(float(cur_pen_bbox[2])), :]
            cur_pen_joints_x = [x - cur_pen_bbox[0] for x in cur_pen_joints_x]
            cur_pen_joints_y = [x - newylow for x in cur_pen_joints_y]
        else:
            longside = cur_pen_bbox[3] - cur_pen_bbox[1]
            shortside = cur_pen_bbox[2] - cur_pen_bbox[0]
            difference = longside - shortside
            newxlow = cur_pen_bbox[0] - difference / 2
            newxhigh = cur_pen_bbox[2] + difference / 2

            if newxlow < 0:
                newxlow = 0
            if newxhigh > cur_img.shape[0]:
                newxhigh = box_size

            cur_img = cur_img[int(float(cur_pen_bbox[1])):int(float(cur_pen_bbox[3])),
                      int(float(newxlow)):int(float(newxhigh)), :]
            cur_pen_joints_x = [x - newylow for x in cur_pen_joints_x]
            cur_pen_joints_y = [x - cur_pen_bbox[1] for x in cur_pen_joints_y]


        # Display joints
        '''
        for i in range(len(cur_pen_joints_x)):
            cv2.circle(cur_img, center=(int(cur_pen_joints_x[i]), int(cur_pen_joints_y[i])),radius=3, color=(255,0,0), thickness=-1)
            cv2.imshow('', cur_img)
            cv2.waitKey(500)
        cv2.imshow('', cur_img)
        cv2.waitKey(1)
        '''
        output_image = np.ones(shape=(box_size, box_size, 3)) * 128
        output_heatmaps = np.zeros((box_size, box_size, num_of_joints))

        # Resize and pad image to fit output image size
        if cur_img.shape[0] > cur_img.shape[1]:
            scale = box_size / (cur_img.shape[0] * 1.0)

            # Relocalize points
            cur_pen_joints_x = map(lambda x: x * scale, cur_pen_joints_x)
            cur_pen_joints_y = map(lambda x: x * scale, cur_pen_joints_y)

            # Resize image
            image = cv2.resize(cur_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
            offset = image.shape[1] % 2

            output_image[:, int(box_size / 2 - math.floor(image.shape[1] / 2)): int(
                box_size / 2 + math.floor(image.shape[1] / 2) + offset), :] = image
            cur_pen_joints_x = map(lambda x: x + (box_size / 2 - math.floor(image.shape[1] / 2)),
                                    cur_pen_joints_x)

            cur_pen_joints_x = np.asarray(cur_pen_joints_x)
            cur_pen_joints_y = np.asarray(cur_pen_joints_y)

            if SHOW_INFO:
                hmap = np.zeros((box_size, box_size))
                # Plot joints
                for i in range(num_of_joints):
                    cv2.circle(output_image, (int(cur_pen_joints_x[i]), int(cur_pen_joints_y[i])), 1, (0, 255, 0), 2)

                    # Generate joint gaussian map
                    part_heatmap= cpm_utils.gaussian_img(box_size,box_size,cur_pen_joints_x[i],cur_pen_joints_y[i],1)
                    #part_heatmap = utils.make_gaussian(output_image.shape[0], gaussian_radius,
                     #                                  [cur_pen_joints_x[i], cur_pen_joints_y[i]])
                    hmap += part_heatmap * 50
            else:
                for i in range(num_of_joints):
                    #output_heatmaps[:, :, i] = utils.make_gaussian(box_size, gaussian_radius,
                    #                                               [cur_pen_joints_x[i], cur_pen_joints_y[i]])
                    output_heatmaps[:, :, i]= cpm_utils.gaussian_img(box_size,box_size,cur_pen_joints_x[i],cur_pen_joints_y[i],1)

        else:
            scale = box_size / (cur_img.shape[1] * 1.0)

            # Relocalize points
            cur_pen_joints_x = list(map(lambda x: x * scale, cur_pen_joints_x))
            cur_pen_joints_y = list(map(lambda x: x * scale, cur_pen_joints_y))

            # Resize image
            image = cv2.resize(cur_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
            offset = image.shape[0] % 2

            output_image[int(box_size / 2 - math.floor(image.shape[0] / 2)): int(
                box_size / 2 + math.floor(image.shape[0] / 2) + offset), :, :] = image
            cur_pen_joints_y = list(map(lambda x: x + (box_size / 2 - math.floor(image.shape[0] / 2)),
                                    cur_pen_joints_y))

            cur_pen_joints_x = np.asarray(cur_pen_joints_x)
            cur_pen_joints_y = np.asarray(cur_pen_joints_y)


            if SHOW_INFO:
                hmap = np.zeros((box_size, box_size))
                # Plot joints
                for i in range(num_of_joints):
                    cv2.circle(output_image, (int(cur_pen_joints_x[i]), int(cur_pen_joints_y[i])), 1, (0, 255, 0), 2)

                    # Generate joint gaussian map
                    part_heatmap = cpm_utils.make_gaussian(output_image.shape[0], gaussian_radius,
                                                       [cur_pen_joints_x[i], cur_pen_joints_y[i]])
                    hmap += part_heatmap * 50
            else:
                for i in range(num_of_joints):
                    output_heatmaps[:, :, i] = cpm_utils.make_gaussian(box_size, gaussian_radius,
                                                                   [cur_pen_joints_x[i], cur_pen_joints_y[i]])
        if SHOW_INFO:
            cv2.imshow('', hmap.astype(np.uint8))
            cv2.imshow('i', output_image.astype(np.uint8))
            cv2.waitKey(0)

        # Create background map
        output_background_map = np.ones((box_size, box_size)) - np.amax(output_heatmaps, axis=2)
        output_heatmaps = np.concatenate((output_heatmaps, output_background_map.reshape((box_size, box_size, 1))),
                                         axis=2)
        #cv2.imshow('', (output_background_map*255).astype(np.uint8))
        #cv2.imshow('h', (np.amax(output_heatmaps[:, :, 0:7], axis=2)*255).astype(np.uint8))
        #cv2.waitKey(1000)

        print(cur_pen_joints_x)
        print(cur_pen_joints_y)

        coords_set = np.concatenate((np.reshape(cur_pen_joints_x, (num_of_joints, 1)),
                                     np.reshape(cur_pen_joints_y, (num_of_joints, 1))),
                                    axis=1)

        print(coords_set)

        output_image_raw = output_image.astype(np.uint8).tostring()
        output_heatmaps_raw = output_heatmaps.flatten().tolist()
        output_coords_raw = coords_set.flatten().tolist()

        raw_sample = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(output_image_raw),
            'keypoints': _float64_feature(output_coords_raw),
        }))
        tfr_writer.write(raw_sample.SerializeToString())

        img_count += 1
        if img_count % 1 == 0:
            print('Processed %d images, took %f seconds' % (img_count, time.time() - t1))
            t1 = time.time()

tfr_writer.close()

writer = dw(box_size)