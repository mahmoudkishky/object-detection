import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt
from common import extract_rois, compute_iou, compute_map, decode_rcnn
import numpy as np
import configs
import os
import random
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

class Trainer():
    def __init__(self, rcnn, training_generator, val_generator, epochs, learning_rate = 5e-4):

        self.rcnn = rcnn
        self.training_generator = training_generator
        self.val_generator = val_generator
        self.optimizer = Adam(learning_rate=learning_rate)
        self.epochs = epochs

    def __compute_loss(self, y_true, y_pred):

        pred_score, pred_bb = y_pred
        gt_scores, gt_bb = y_true

        label_loss = tf.keras.losses.categorical_crossentropy(gt_scores, pred_score)

        # background ROIs have an all-zero box target and must not train the regressor
        has_box = tf.cast(tf.reduce_any(tf.not_equal(gt_bb, 0.0), axis=-1), tf.float32)
        bbox_loss = tf.reduce_mean(tf.abs(gt_bb - pred_bb), axis=-1) * has_box

        rcnn_loss = label_loss + 0.5 * bbox_loss
        return rcnn_loss

    def __image_loss(self, image, bboxes, training):
        """Mean loss over the sampled ROIs of one image, or None when no ROI was sampled."""

        crops, all_scores, all_bb = [], [], []

        with tf.device('/CPU:0'):
            # the data generator yields float images in [0, 1], selective search needs uint8
            rois = extract_rois(np.clip(image * 255.0, 0, 255).astype(np.uint8))

        for i, (x, y, w, h) in enumerate(rois[:configs.ROIS_PER_IMAGE]):

            # to compute to loss we have to build the ground truth annotations
            roi_bb = [x, y, x + w, y + h]
            best_iou = 0
            best_bb = None

            for bbox in bboxes:
                iou = compute_iou(roi_bb, bbox[1])
                if iou > best_iou:
                    best_iou = iou
                    best_iou_scores = bbox[0]
                    best_bb = bbox[1]

            if best_iou > 0.5:
                gt_scores = best_iou_scores
                gt_bb = best_bb

            # to avoid unbalanced learning, keep only 25% of background ROIs
            elif random.random() < 0.25:
                    gt_scores = tf.one_hot(configs.VOC_LABELS['background'], depth=len(configs.VOC_LABELS))
                    gt_bb = None
            else:
                continue

            if gt_bb is not None:

                xg_min, yg_min, xg_max, yg_max = gt_bb

                wg = xg_max - xg_min
                hg = yg_max - yg_min
                xg = xg_min + wg / 2
                yg = yg_min + hg / 2

                xp = x + w / 2
                yp = y + h / 2
                wp = w
                hp = h

                tx = (xg - xp) / wp
                ty = (yg - yp) / hp
                tw = np.log(wg / wp)
                th = np.log(hg / hp)

                gt_bb = tf.convert_to_tensor([tx, ty, tw, th], dtype=tf.float32)
            else:
                gt_bb = tf.convert_to_tensor([0., 0., 0., 0.], dtype=tf.float32)

            cropped_image = image[y:y+h, x:x+w]
            cropped_image = cv2.resize(cropped_image, dsize = (configs.IMAGE_SIZE_X, configs.IMAGE_SIZE_Y), interpolation = cv2.INTER_CUBIC)

            crops.append(cropped_image)
            all_scores.append(gt_scores)
            all_bb.append(gt_bb)

        if not crops:
            return None

        # all sampled ROIs of the image go through the network in a single batch
        y_true = [tf.stack([tf.cast(s, tf.float32) for s in all_scores]), tf.stack(all_bb)]
        y_pred = self.rcnn(tf.convert_to_tensor(np.asarray(crops, dtype=np.float32)), training=training)

        return tf.reduce_mean(self.__compute_loss(y_true, y_pred))

    def __train_step(self, training_batch):

        with tf.GradientTape() as gt:

            image_loss = []

            for (image, bboxes) in zip(training_batch[0], training_batch[1]):

                loss = self.__image_loss(image, bboxes, training=True)
                if loss is not None:
                    image_loss.append(loss)

            if not image_loss:
                return 0.0

            batch_loss = tf.reduce_mean(image_loss)

        gradients = gt.gradient(batch_loss, self.rcnn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.rcnn.trainable_variables))

        return batch_loss

    def __detect(self, image):
        with tf.device('/CPU:0'):
            rois = extract_rois(np.clip(image * 255.0, 0, 255).astype(np.uint8))
        rois = [tuple(int(v) for v in roi) for roi in rois[:configs.ROIS_PER_IMAGE] if roi[2] > 0 and roi[3] > 0]
        if not rois:
            return [], [], []

        crops = np.asarray([
            cv2.resize(image[y:y+h, x:x+w], dsize = (configs.IMAGE_SIZE_X, configs.IMAGE_SIZE_Y), interpolation = cv2.INTER_CUBIC)
            for x, y, w, h in rois
        ], dtype=np.float32)
        class_scores, bbox_preds = self.rcnn(tf.convert_to_tensor(crops), training=False)

        return decode_rcnn(
            rois, class_scores.numpy(), bbox_preds.numpy(), configs.VOC_LABELS['background'],
            configs.IMAGE_SIZE_X, configs.IMAGE_SIZE_Y,
        )

    def val_metric(self):
        """Run the validation set; return (mean loss, mAP@0.5)."""

        losses, detections, ground_truths = [], [], []

        num_batches = len(self.val_generator)
        if configs.BATCH_SIZE is not None:
            num_batches = min(num_batches, configs.BATCH_SIZE)

        for batch_idx in range(num_batches):
            images, objects = self.val_generator[batch_idx]

            for image, bboxes in zip(images, objects):

                loss = self.__image_loss(image, bboxes, training=False)
                if loss is not None:
                    losses.append(float(loss))

                detections.append(self.__detect(image))
                ground_truths.append((
                    [list(bbox) for _, bbox in bboxes],
                    [int(np.argmax(label)) for label, _ in bboxes],
                ))

        mean_loss = float(np.mean(losses)) if losses else 0.0
        map_score, _ = compute_map(detections, ground_truths, len(configs.VOC_LABELS))
        return mean_loss, map_score

    def train(self):

        for epoch in range(self.epochs):

            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            for step, training_batch in enumerate(self.training_generator):

                loss = self.__train_step(training_batch)
                if step % configs.LOG_EVERY_STEP == 0:
                    print(f"step {step}: training_loss = {float(loss):.4f}")

            val_loss, map_score = self.val_metric()
            print(f"val_loss = {val_loss:.4f}, mAP@0.5 = {map_score:.4f}")

            if (epoch + 1) % configs.SAVE_EVERY_EPOCH == 0:
                self.training_generator.on_epoch_end()
                self.rcnn.save(f"checkpoints/rcnn/epoch_{epoch + 1}")
                print(f"Saved parameters at checkpoints/rcnn/epoch_{epoch + 1}")
