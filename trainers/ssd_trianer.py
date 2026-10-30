import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import configs
from common import compute_iou, compute_map

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class Trainer():

    def __init__(self, ssd, training_generator, val_generator, epochs, learning_rate=5e-4):

        self.ssd = ssd
        self.training_generator = training_generator
        self.val_generator = val_generator
        self.optimizer = Adam(learning_rate=learning_rate)
        self.epochs = epochs

        # SSD-style training weights. This is closer to the official SSD objective than
        # the earlier generic box/class weighting, while still matching the simplified
        # feature-map layout of this project.
        self.loc_weight = 1.0
        self.neg_ratio = 3.0
        self.background_class_id = int(configs.VOC_LABELS.get('background', self.ssd.num_classes - 1))
        if self.background_class_id < 0 or self.background_class_id >= self.ssd.num_classes:
            raise ValueError(
                f"Invalid background_class_id={self.background_class_id} for num_classes={self.ssd.num_classes}"
            )

        # Simple SSD-style scales per feature map.
        # The boxes are centered around the cell's corresponding location. Their dimensions are predefined using two parameters:
        # - Scale S_k: controls the box's size.
        # - Aspect ratio r = w/h: controls whether the box is wide or tall.
        # A common construction is:
        # w=S_k*sqrt(r), h=S_k/sqrt(r)
    
        self.predictor_scales = [0.08, 0.16, 0.32]
        self.predictor_aspect_ratios = [
            [1.0, 2.0, 0.5, 1.0],
            [1.0, 2.0, 0.5, 3.0, 1.0, 1 / 3],
            [1.0, 2.0, 0.5, 3.0, 1.0, 1 / 3],
        ]

        self.default_boxes = []
        # for each feature map, generate default boxes for each cell and each aspect ratio
        # per cell in the feature map, we have a set of default boxes defined by the predictor scales and aspect ratios.
        # the bb are defined by the center coordinates (cx, cy) and the width and height (rw, rh) derived from the scale and aspect ratio.

        for layer_idx, (h, w) in enumerate(self.ssd.feature_map_sizes):
            boxes = []
            for gy in range(h):
                for gx in range(w):
                    cx = (gx + 0.5) / w
                    cy = (gy + 0.5) / h
                    for ratio in self.predictor_aspect_ratios[layer_idx][:self.ssd.num_anchors_per_layer[layer_idx]]:
                        rw = self.predictor_scales[layer_idx] * np.sqrt(ratio)
                        rh = self.predictor_scales[layer_idx] / np.sqrt(ratio)
                        x_min = cx - rw / 2.0
                        y_min = cy - rh / 2.0
                        x_max = cx + rw / 2.0
                        y_max = cy + rh / 2.0
                        boxes.append([x_min, y_min, x_max, y_max])
            self.default_boxes.append(np.array(boxes, dtype=np.float32).reshape(h, w, self.ssd.num_anchors_per_layer[layer_idx], 4))

        # return a list of default boxes for each feature map layer
        # the default boxes size is:
        # map_x * map_y * num_anchors_per_layer[layer_idx] * 4
        # eg. feature map layer 1: 56 * 56 * 4 * 4 = 50176
        # eg. feature map layer 2: 28 * 28 * 6 * 4 = 18816
        # eg. feature map layer 3: 14 * 14 * 6 * 4 = 4704

    def __ssd_loss(self, y_true, y_pred):
        loc_loss_all = []
        conf_loss_all = []
        pos_mask_all = []
        neg_mask_all = []

        for layer_idx, (true_layer, pred_layer) in enumerate(zip(y_true, y_pred)):
            h, w = self.ssd.feature_map_sizes[layer_idx]
            anchors = self.ssd.num_anchors_per_layer[layer_idx]

            true_layer = tf.reshape(true_layer, (-1, h, w, anchors, 4 + self.ssd.num_classes))
            pred_layer = tf.reshape(pred_layer, (-1, h, w, anchors, 4 + self.ssd.num_classes))

            class_targets = true_layer[..., 4:]
            foreground_targets = tf.concat(
                [
                    class_targets[..., :self.background_class_id],
                    class_targets[..., self.background_class_id + 1:],
                ],
                axis=-1,
            )
            background_targets = class_targets[..., self.background_class_id:self.background_class_id + 1]

            pos_mask = tf.reduce_any(foreground_targets > 0.5, axis=-1)
            neg_mask = tf.squeeze(background_targets > 0.5, axis=-1)

            loc_pred = pred_layer[..., :4]
            loc_true = true_layer[..., :4]
            diff = tf.abs(loc_pred - loc_true)
            loc_loss = tf.reduce_sum(
                self.loc_weight * tf.where(diff < 1.0, 0.5 * tf.square(diff), diff - 0.5),
                axis=-1,
            )

            class_logits = pred_layer[..., 4:]
            conf_loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=class_targets,
                logits=class_logits,
            )

            loc_loss_all.append(tf.reshape(loc_loss, [tf.shape(loc_loss)[0], -1]))
            conf_loss_all.append(tf.reshape(conf_loss, [tf.shape(conf_loss)[0], -1]))
            pos_mask_all.append(tf.reshape(pos_mask, [tf.shape(pos_mask)[0], -1]))
            neg_mask_all.append(tf.reshape(neg_mask, [tf.shape(neg_mask)[0], -1]))

        loc_loss_all = tf.concat(loc_loss_all, axis=1)
        conf_loss_all = tf.concat(conf_loss_all, axis=1)
        pos_mask_all = tf.concat(pos_mask_all, axis=1)
        neg_mask_all = tf.concat(neg_mask_all, axis=1)

        pos_mask_f = tf.cast(pos_mask_all, tf.float32)
        num_pos = tf.reduce_sum(pos_mask_f, axis=1)

        positive_conf = tf.reduce_sum(conf_loss_all * pos_mask_f, axis=1)

        # Hard-negative mining: keep only top-k background anchors per image,
        # where k = neg_ratio * num_pos (clipped by available negatives).
        neg_conf_for_sort = tf.where(
            neg_mask_all,
            conf_loss_all,
            tf.fill(tf.shape(conf_loss_all), tf.constant(-1e6, dtype=conf_loss_all.dtype)),
        )
        sorted_indices = tf.argsort(neg_conf_for_sort, axis=1, direction='DESCENDING')
        ranks = tf.argsort(sorted_indices, axis=1)

        neg_available = tf.reduce_sum(tf.cast(neg_mask_all, tf.int32), axis=1)
        max_neg = tf.cast(self.neg_ratio * num_pos, tf.int32)
        num_neg = tf.minimum(neg_available, max_neg)
        hard_neg_mask = tf.logical_and(neg_mask_all, ranks < tf.expand_dims(num_neg, axis=1))
        hard_neg_mask_f = tf.cast(hard_neg_mask, tf.float32)
        negative_conf = tf.reduce_sum(conf_loss_all * hard_neg_mask_f, axis=1)

        loc_loss = tf.reduce_sum(loc_loss_all * pos_mask_f, axis=1)

        normalizer = tf.maximum(num_pos, 1.0)
        sample_loss = (loc_loss + positive_conf + negative_conf) / normalizer

        return tf.reduce_mean(sample_loss)

    def __build_target_labels(self, images, objects):
        batch_size = len(images)
        y_true = []

        for layer_idx, (h, w) in enumerate(self.ssd.feature_map_sizes):
            anchors = self.ssd.num_anchors_per_layer[layer_idx]
            loc_targets = tf.zeros((batch_size, h, w, anchors, 4), dtype=tf.float32)
            class_targets = tf.one_hot(
                tf.fill((batch_size, h, w, anchors), self.background_class_id),
                depth=self.ssd.num_classes,
                dtype=tf.float32,
            )
            y_true.append(tf.concat([loc_targets, class_targets], axis=-1))

        for image_idx in range(batch_size):
            for obj in objects[image_idx]:
                label, bbox = obj

                x_min, y_min, x_max, y_max = bbox
                x_c = ((x_min + x_max) / 2.0) / configs.IMAGE_SIZE_X
                y_c = ((y_min + y_max) / 2.0) / configs.IMAGE_SIZE_Y
                w = (x_max - x_min) / configs.IMAGE_SIZE_X
                h = (y_max - y_min) / configs.IMAGE_SIZE_Y
                gt_box = [x_c, y_c, w, h]
                gt_box_xyxy = [
                    x_min / configs.IMAGE_SIZE_X,
                    y_min / configs.IMAGE_SIZE_Y,
                    x_max / configs.IMAGE_SIZE_X,
                    y_max / configs.IMAGE_SIZE_Y,
                ]

                if isinstance(label, (list, tuple, np.ndarray, tf.Tensor)):
                    class_vector = tf.cast(label, tf.float32)
                else:
                    class_vector = tf.one_hot(int(label), depth=self.ssd.num_classes, dtype=tf.float32)

                class_vector = tf.reshape(class_vector, [-1])
                class_vector = tf.concat(
                    [class_vector, tf.zeros([self.ssd.num_classes], dtype=tf.float32)],
                    axis=0,
                )[:self.ssd.num_classes]

                # GT objects must be foreground; keep background channel at zero.
                bg_mask = tf.one_hot(self.background_class_id, depth=self.ssd.num_classes, dtype=tf.float32)
                class_vector = class_vector * (1.0 - bg_mask)
                class_sum = tf.reduce_sum(class_vector)
                if float(class_sum.numpy()) <= 0.0:
                    continue
                class_vector = class_vector / class_sum

                best_layer = None
                best_cell = None
                best_anchor = None
                best_iou = -1.0

                for layer_idx, layer_boxes in enumerate(self.default_boxes):
                    h_layer, w_layer, anchors, _ = layer_boxes.shape
                    for gy in range(h_layer):
                        for gx in range(w_layer):
                            for anchor_idx in range(anchors):
                                anchor_box = layer_boxes[gy, gx, anchor_idx]
                                iou = compute_iou(gt_box_xyxy, anchor_box, box_format='xyxy')
                                if iou > best_iou:
                                    best_iou = iou
                                    best_layer = layer_idx
                                    best_cell = (gy, gx)
                                    best_anchor = anchor_idx

                if best_layer is None:
                    continue

                gy, gx = best_cell
                default_box = self.default_boxes[best_layer][gy, gx, best_anchor]
                default_cx = (default_box[0] + default_box[2]) / 2.0
                default_cy = (default_box[1] + default_box[3]) / 2.0
                default_w = max(default_box[2] - default_box[0], 1e-6)
                default_h = max(default_box[3] - default_box[1], 1e-6)

                gt_cx, gt_cy, gt_w, gt_h = gt_box
                target_box = [
                    (gt_cx - default_cx) / default_w,
                    (gt_cy - default_cy) / default_h,
                    np.log(gt_w / default_w),
                    np.log(gt_h / default_h),
                ]

                indices = [
                    [image_idx, gy, gx, best_anchor, 0],
                    [image_idx, gy, gx, best_anchor, 1],
                    [image_idx, gy, gx, best_anchor, 2],
                    [image_idx, gy, gx, best_anchor, 3],
                ]
                values = tf.constant(target_box, dtype=tf.float32)
                y_true[best_layer] = tf.tensor_scatter_nd_update(y_true[best_layer], indices, values)

                class_indices = [
                    [image_idx, gy, gx, best_anchor, 4 + j]
                    for j in range(self.ssd.num_classes)
                ]
                y_true[best_layer] = tf.tensor_scatter_nd_update(
                    y_true[best_layer],
                    class_indices,
                    tf.reshape(class_vector, [-1]),
                )

        return y_true

    @staticmethod
    def __softmax(x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / np.sum(e, axis=-1, keepdims=True)

    def __decode_predictions(self, y_pred, image_idx, threshold=0.05):
        """Decode one image's SSD outputs into pixel boxes, mirroring the encoding in __build_target_labels."""
        boxes, scores, labels = [], [], []
        for layer_idx, layer_pred in enumerate(y_pred):
            h, w = self.ssd.feature_map_sizes[layer_idx]
            anchors = self.ssd.num_anchors_per_layer[layer_idx]
            cells = tf.reshape(layer_pred[image_idx], (h, w, anchors, 4 + self.ssd.num_classes)).numpy()
            default = self.default_boxes[layer_idx]

            probs = self.__softmax(cells[..., 4:])
            probs[..., self.background_class_id] = 0.0
            class_ids = np.argmax(probs, axis=-1)
            conf = np.max(probs, axis=-1)
            keep = conf >= threshold
            if not np.any(keep):
                continue

            d = default[keep]
            d_cx, d_cy = (d[:, 0] + d[:, 2]) / 2.0, (d[:, 1] + d[:, 3]) / 2.0
            d_w, d_h = np.maximum(d[:, 2] - d[:, 0], 1e-6), np.maximum(d[:, 3] - d[:, 1], 1e-6)
            loc = cells[..., :4][keep]
            cx, cy = d_cx + loc[:, 0] * d_w, d_cy + loc[:, 1] * d_h
            bw, bh = np.exp(loc[:, 2]) * d_w, np.exp(loc[:, 3]) * d_h

            boxes.append(np.stack([
                np.clip(cx - bw / 2, 0.0, 1.0) * configs.IMAGE_SIZE_X,
                np.clip(cy - bh / 2, 0.0, 1.0) * configs.IMAGE_SIZE_Y,
                np.clip(cx + bw / 2, 0.0, 1.0) * configs.IMAGE_SIZE_X,
                np.clip(cy + bh / 2, 0.0, 1.0) * configs.IMAGE_SIZE_Y,
            ], axis=-1))
            scores.append(conf[keep])
            labels.append(class_ids[keep])

        if not boxes:
            return [], [], []
        boxes = np.concatenate(boxes).astype(np.float32)
        scores = np.concatenate(scores).astype(np.float32)
        labels = np.concatenate(labels)
        selected = tf.image.non_max_suppression(boxes, scores, max_output_size=50, iou_threshold=0.5).numpy()
        return boxes[selected].tolist(), scores[selected].tolist(), labels[selected].tolist()

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

            y_pred = self.ssd(tf.convert_to_tensor(images), training=False)
            y_true = self.__build_target_labels(images, objects)
            losses.append(float(self.__ssd_loss(y_true, y_pred)))

            for image_idx in range(len(images)):
                detections.append(self.__decode_predictions(y_pred, image_idx))
                ground_truths.append((
                    [list(bbox) for _, bbox in objects[image_idx]],
                    [self.__label_to_id(label) for label, _ in objects[image_idx]],
                ))

        mean_loss = float(np.mean(losses)) if losses else 0.0
        map_score, _ = compute_map(detections, ground_truths, self.ssd.num_classes)
        return mean_loss, map_score

    def __train_step(self, training_batch):
        images, objects = training_batch

        with tf.GradientTape() as gt:
            y_pred = self.ssd(tf.convert_to_tensor(images), training=True)
            y_true = self.__build_target_labels(images, objects)
            batch_loss = self.__ssd_loss(y_true, y_pred)

        gradients = gt.gradient(batch_loss, self.ssd.trainable_variables)
        gradients = [g for g in gradients if g is not None]
        self.optimizer.apply_gradients(zip(gradients, [v for v in self.ssd.trainable_variables if v is not None]))

        return batch_loss

    def train(self):
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            for step, training_batch in enumerate(self.training_generator):
                loss = self.__train_step(training_batch)
                if step % configs.LOG_EVERY_STEP == 0:
                    print(f"step {step}: training_loss = {loss:.4f}")

            val_loss, map_score = self.val_metric()
            print(f"val_loss = {val_loss:.4f}, mAP@0.5 = {map_score:.4f}")

            if (epoch + 1) % configs.SAVE_EVERY_EPOCH == 0:
                self.training_generator.on_epoch_end()
                self.ssd.save(f"checkpoints/ssd/epoch_{epoch + 1}")
                print(f"Saved parameters at checkpoints/ssd/epoch_{epoch + 1}")
