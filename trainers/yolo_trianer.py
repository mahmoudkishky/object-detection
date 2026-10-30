import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import cv2
import matplotlib.pyplot as plt
import numpy as np
import configs
import os
import random
from common import compute_iou, compute_map
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Trainer():
    def __init__(self, yolo, training_generator, val_generator, epochs, learning_rate = 5e-4):

        self.yolo = yolo
        self.training_generator = training_generator
        self.val_generator = val_generator
        self.optimizer = Adam(learning_rate=learning_rate)
        self.epochs = epochs

    def __yolo_loss(self, y_true, y_pred):
        """YOLO-style loss using the current y_true/yolo_predictions structure."""
        y_pred = tf.reshape(
            y_pred,
            (-1, self.yolo.grid_size[0], self.yolo.grid_size[1], self.yolo.num_anchors, 5 + self.yolo.num_classes)
        )
        y_true = tf.reshape(
            y_true,
            (-1, self.yolo.grid_size[0], self.yolo.grid_size[1], self.yolo.num_anchors, 5 + self.yolo.num_classes)
        )

        obj_mask = tf.cast(y_true[..., 4:5] > 0.5, tf.float32)
        noobj_mask = 1.0 - obj_mask

        box_loss = tf.reduce_sum(obj_mask * tf.square(y_pred[..., :4] - y_true[..., :4]), axis=[1, 2, 3, 4])
        obj_loss = tf.reduce_sum(obj_mask * tf.square(y_pred[..., 4:5] - y_true[..., 4:5]), axis=[1, 2, 3, 4])
        noobj_loss = tf.reduce_sum(noobj_mask * tf.square(y_pred[..., 4:5] - y_true[..., 4:5]), axis=[1, 2, 3, 4])

        class_loss = tf.reduce_sum(
            obj_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true[..., 5:], logits=y_pred[..., 5:]),
            axis=[1, 2, 3, 4]
        )

        loss = 5.0 * box_loss + obj_loss + 0.5 * noobj_loss + class_loss
        return tf.reduce_mean(loss)



    def __build_targets(self, objects, batch_size):
        target_shape = (
            batch_size,
            self.yolo.grid_size[0],
            self.yolo.grid_size[1],
            self.yolo.num_anchors,
            5 + self.yolo.num_classes,
        )
        yolo_true = tf.zeros(target_shape, dtype=tf.float32)

        for i in range(batch_size):
            for obj in objects[i]:
                label, bbox = obj
                x_c = (bbox[0] + bbox[2]) / (2 * configs.IMAGE_SIZE_X)
                y_c = (bbox[1] + bbox[3]) / (2 * configs.IMAGE_SIZE_Y)
                w = (bbox[2] - bbox[0]) / configs.IMAGE_SIZE_X
                h = (bbox[3] - bbox[1]) / configs.IMAGE_SIZE_Y
                normalized_bbox = [x_c, y_c, w, h]

                grid_x = min(int(x_c * self.yolo.grid_size[1]), self.yolo.grid_size[1] - 1)
                grid_y = min(int(y_c * self.yolo.grid_size[0]), self.yolo.grid_size[0] - 1)

                best_iou = 0.0
                best_anchor = 0
                for anchor_idx, anchor in enumerate(self.yolo.anchors):
                    anchor_box = [
                        (grid_x + 0.5) / self.yolo.grid_size[1],
                        (grid_y + 0.5) / self.yolo.grid_size[0],
                        anchor[0] / self.yolo.grid_size[1],
                        anchor[1] / self.yolo.grid_size[0],
                    ]
                    iou = compute_iou(normalized_bbox, anchor_box, box_format='cxcywh')
                    if iou > best_iou:
                        best_iou = iou
                        best_anchor = anchor_idx

                base_index = [i, grid_y, grid_x, best_anchor]
                yolo_true = tf.tensor_scatter_nd_update(
                    yolo_true,
                    [base_index + [offset] for offset in range(4)],
                    normalized_bbox,
                )
                yolo_true = tf.tensor_scatter_nd_update(
                    yolo_true,
                    [base_index + [4]],
                    [1.0],
                )
                if isinstance(label, (list, tuple, np.ndarray, tf.Tensor)):
                    class_vector = tf.cast(label, tf.float32)
                else:
                    class_vector = tf.one_hot(int(label), depth=self.yolo.num_classes)
                yolo_true = tf.tensor_scatter_nd_update(
                    yolo_true,
                    [base_index + [5 + j] for j in range(self.yolo.num_classes)],
                    tf.reshape(class_vector, [-1]),
                )

        return yolo_true

    def __decode_prediction(self, prediction, threshold=0.05):
        """Decode one image's output into pixel boxes, mirroring the encoding used by __build_targets."""
        reshaped = tf.reshape(
            prediction,
            (self.yolo.grid_size[0], self.yolo.grid_size[1], self.yolo.num_anchors, 5 + self.yolo.num_classes),
        ).numpy()

        objectness = np.clip(reshaped[..., 4], 0.0, 1.0)
        class_probs = 1.0 / (1.0 + np.exp(-reshaped[..., 5:]))
        class_ids = np.argmax(class_probs, axis=-1)
        scores = objectness * np.max(class_probs, axis=-1)

        keep = scores >= threshold
        if not np.any(keep):
            return [], [], []

        cx, cy, w, h = [reshaped[..., i][keep] for i in range(4)]
        boxes = np.stack([
            np.clip(cx - w / 2, 0.0, 1.0) * configs.IMAGE_SIZE_X,
            np.clip(cy - h / 2, 0.0, 1.0) * configs.IMAGE_SIZE_Y,
            np.clip(cx + w / 2, 0.0, 1.0) * configs.IMAGE_SIZE_X,
            np.clip(cy + h / 2, 0.0, 1.0) * configs.IMAGE_SIZE_Y,
        ], axis=-1).astype(np.float32)
        scores, class_ids = scores[keep].astype(np.float32), class_ids[keep]

        selected = tf.image.non_max_suppression(boxes, scores, max_output_size=50, iou_threshold=0.5).numpy()
        return boxes[selected].tolist(), scores[selected].tolist(), class_ids[selected].tolist()

    @staticmethod
    def __label_to_id(label):
        if isinstance(label, (list, tuple, np.ndarray, tf.Tensor)):
            return int(np.argmax(np.asarray(label)))
        return int(label)

    def val_metric(self):
        """Run the validation set; return (mean loss, mAP@0.5)."""
        losses, detections, ground_truths = [], [], []

        for batch_idx in range(len(self.val_generator)):
            images, objects = self.val_generator[batch_idx]
            if len(images) == 0:
                continue

            predictions = self.yolo(tf.convert_to_tensor(images), training=False)
            targets = self.__build_targets(objects, len(images))
            losses.append(float(self.__yolo_loss(targets, predictions)))

            for image_idx, prediction in enumerate(predictions):
                detections.append(self.__decode_prediction(prediction))
                ground_truths.append((
                    [list(bbox) for _, bbox in objects[image_idx]],
                    [self.__label_to_id(label) for label, _ in objects[image_idx]],
                ))

        mean_loss = float(np.mean(losses)) if losses else 0.0
        map_score, _ = compute_map(detections, ground_truths, self.yolo.num_classes)
        return mean_loss, map_score

    def __train_step(self, training_batch):

        # the yolo predictions are expected to be in format:
        # [grid[0, 0] prediction, grid[0, 1] prediction, ....,grid[0, 6] prediction,
        #  grid[1, 0] prediction, ..., grid[6, 6] prediction]
        # each prediction contains 30 values as follows:
        # anchor 0 -> [class probabilities (20), objectness score (1), bounding box coordinates (4)]
        # anchor 1 -> [class probabilities (20), objectness score (1), bounding box coordinates (4)]
        # we expect the true lables to be in the same format as the predictions,
        # for class probabilities and objectness score, we use categorical crossentropy loss
        # for bounding box coordinates, we use mean squared error loss
        # training data generator produce a batch of(image, objects)
        # each object contains class label, and bounding box coordinates (label, [x_min, y_min, x_max, y_max])

        images, objects = training_batch

        with tf.GradientTape() as gt:
            yolo_predictions = self.yolo(tf.convert_to_tensor(images), training=True)
            yolo_predictions = tf.reshape(
                yolo_predictions,
                (-1, self.yolo.grid_size[0], self.yolo.grid_size[1], self.yolo.num_anchors, 5 + self.yolo.num_classes),
            )

            yolo_true = self.__build_targets(objects, len(images))

            batch_loss = self.__yolo_loss(yolo_true, yolo_predictions)

        gradients = gt.gradient(batch_loss, self.yolo.trainable_variables)
        gradients = [g for g in gradients if g is not None]
        self.optimizer.apply_gradients(zip(gradients, [v for v in self.yolo.trainable_variables if v is not None]))

        return batch_loss
    
    def train(self):

        for epoch in range(self.epochs):

            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            for step, training_batch in enumerate(self.training_generator):

                loss = self.__train_step(training_batch)
                if step % configs.LOG_EVERY_STEP == 0:
                    print(
                        f"step {step}: training_loss = {float(loss):.4f}"
                    )

            val_loss, map_score = self.val_metric()
            print(f"val_loss = {val_loss:.4f}, mAP@0.5 = {map_score:.4f}")

            if (epoch + 1) % configs.SAVE_EVERY_EPOCH == 0:
                self.training_generator.on_epoch_end()
                self.yolo.save(f"checkpoints/yolo/epoch_{epoch + 1}")
                print(f"Saved parameters at checkpoints/yolo/epoch_{epoch + 1}")
