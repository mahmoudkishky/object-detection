from math import floor
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import Sequence
import xml.etree.ElementTree as ET
import os
import cv2
import matplotlib.pyplot as plt
import configs


class VOCDataGenerator(Sequence):
    def __init__(self, data_path, voc_labels, batch_size, isTraining):
        
        self.data_path = data_path
        self.batch_size = batch_size

        if isTraining:
            idFile = configs.TRAIN_FILE
        else:
            idFile = configs.VAL_FILE

        self.data_ids = []
        with open(os.path.join(self.data_path, 'ImageSets/Main/', idFile)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 0:
                    continue
                image_id = parts[0]
                if len(parts) > 1:
                    # person_train.txt uses 1 / -1 labels for presence of person
                    if parts[1] == '-1':
                        continue
                self.data_ids.append(image_id)

        self.voc_labels = voc_labels
            
    def __len__(self):
        return int(np.ceil(len(self.data_ids) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch_ids = self.data_ids[start:end]

        return self.__fetch_batch(batch_ids)

    def on_epoch_end(self):
        np.random.shuffle(self.data_ids)
    
    def __parse_xml(self, xml_file):

        tree = ET.parse(xml_file)
        root = tree.getroot()

        bboxes = []

        for obj in root.findall("object"):
            
            name = obj.find("name").text
            if name not in self.voc_labels:
                continue

            bbox = obj.find("bndbox")
            one_hot_label = tf.one_hot(self.voc_labels[name], depth=len(self.voc_labels)).numpy()

            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            bboxes.append((one_hot_label, [xmin, ymin, xmax, ymax]))

        return bboxes
    

    def __fetch_batch(self, batch_ids):
        
        images = []
        objects = []
        for id in batch_ids:
            annotation_path = os.path.join(self.data_path, 'Annotations', id + '.xml')
            image_path = os.path.join(self.data_path, 'JPEGImages', id + '.jpg')

            if not os.path.exists(annotation_path) or not os.path.exists(image_path):
                continue

            bboxes = self.__parse_xml(annotation_path)
            im = np.array(cv2.imread(image_path))
            if im is None:
                continue

            scaled_objects = []
            for bbox in bboxes:
                label, bb = bbox
                # voco dataset contain images of varying sizes,
                # so we need to scale the bounding boxes to a fixed size
                scale_x = float(configs.IMAGE_SIZE_X / im.shape[1])
                scale_y = float(configs.IMAGE_SIZE_Y / im.shape[0])
                x_min = floor(bb[0] * scale_x)
                y_min = floor(bb[1] * scale_y)
                x_max = floor(bb[2] * scale_x)
                y_max = floor(bb[3] * scale_y)
                scaled_objects.append((label, [x_min, y_min, x_max, y_max]))

            if not scaled_objects:
                continue

            im = cv2.resize(im, dsize=(configs.IMAGE_SIZE_X, configs.IMAGE_SIZE_Y), interpolation=cv2.INTER_CUBIC)
            im = im.astype(np.float32) / 255.0
            images.append(im)
            objects.append(scaled_objects)

        return images, objects

