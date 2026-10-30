import argparse
import importlib
import os
import sys

import cv2
import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

MODEL_CHOICES = ('yolo', 'ssd', 'rcnn')

# Set by load_modules() once --model is known.
configs = None
models = None


def load_modules(model_name):
    global configs, models
    configs = importlib.import_module(f'configs.{model_name}_configs')
    sys.modules['configs'] = configs
    models = importlib.import_module(f'models.{model_name}_model')


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float32)
    logits = logits - np.max(logits)
    exps = np.exp(logits)
    return exps / np.sum(exps)


def nms(boxes, scores, labels, max_output_size, iou_threshold=0.5):
    if not boxes:
        return [], [], []
    indices = tf.image.non_max_suppression(
        boxes=tf.constant(boxes, dtype=tf.float32),
        scores=tf.constant(scores, dtype=tf.float32),
        max_output_size=max_output_size,
        iou_threshold=iou_threshold,
    ).numpy()
    return [boxes[i] for i in indices], [scores[i] for i in indices], [labels[i] for i in indices]


def decode_yolo(prediction, model, threshold=0.2):
    """Convert a flattened YOLO output into pixel-space boxes and scores."""
    grid_h, grid_w = configs.GRID_SIZE
    num_classes = len(configs.VOC_LABELS)

    reshaped = tf.reshape(
        prediction[0],
        (grid_h, grid_w, len(configs.YOLO_ANCHORS), 5 + num_classes),
    ).numpy()

    boxes, scores, labels = [], [], []
    for gy in range(grid_h):
        for gx in range(grid_w):
            for anchor_idx, anchor in enumerate(configs.YOLO_ANCHORS):
                cell = reshaped[gy, gx, anchor_idx]
                objectness = sigmoid(float(cell[4]))
                if objectness < threshold:
                    continue

                tx, ty, tw, th = cell[:4]
                bx = sigmoid(float(tx))
                by = sigmoid(float(ty))
                bw = np.exp(float(tw)) * anchor[0]
                bh = np.exp(float(th)) * anchor[1]

                center_x = ((gx + bx) / grid_w) * configs.IMAGE_SIZE_X
                center_y = ((gy + by) / grid_h) * configs.IMAGE_SIZE_Y
                width = bw * configs.IMAGE_SIZE_X / grid_w
                height = bh * configs.IMAGE_SIZE_Y / grid_h

                class_logits = cell[5:]
                class_id = int(np.argmax(class_logits))
                final_score = objectness * sigmoid(float(class_logits[class_id]))
                if final_score < threshold:
                    continue

                boxes.append([
                    max(0, center_x - width / 2),
                    max(0, center_y - height / 2),
                    min(configs.IMAGE_SIZE_X, center_x + width / 2),
                    min(configs.IMAGE_SIZE_Y, center_y + height / 2),
                ])
                scores.append(float(final_score))
                labels.append(class_id)

    return nms(boxes, scores, labels, max_output_size=20)


def decode_ssd(predictions, model, threshold=0.2):
    """Convert SSD outputs into boxes (pixel space of the resized input), scores, and labels."""
    num_classes = int(getattr(model, 'num_classes', len(configs.VOC_LABELS)))
    if num_classes <= 0:
        num_classes = len(configs.VOC_LABELS)
    background_class_id = int(configs.VOC_LABELS.get('background', num_classes - 1))

    feature_map_sizes = [(37, 37), (19, 19), (10, 10)][:len(predictions)]
    num_anchors_per_layer = [4, 6, 6][:len(predictions)]
    default_scales = [0.08, 0.16, 0.32]
    default_aspect_ratios = [
        [1.0, 2.0, 0.5, 1.0],
        [1.0, 2.0, 0.5, 3.0, 1.0, 1.0 / 3.0],
        [1.0, 2.0, 0.5, 3.0, 1.0, 1.0 / 3.0],
    ]

    boxes, scores, labels = [], [], []
    for layer_idx, pred_layer in enumerate(predictions):
        h, w = feature_map_sizes[layer_idx]
        anchors = num_anchors_per_layer[layer_idx]
        reshaped = tf.reshape(pred_layer[0], (h, w, anchors, 4 + num_classes)).numpy()

        for gy in range(h):
            for gx in range(w):
                for anchor_idx in range(anchors):
                    cell = reshaped[gy, gx, anchor_idx]
                    tx, ty, tw, th = cell[:4]
                    probs = softmax(cell[4:])
                    if 0 <= background_class_id < probs.shape[0]:
                        probs[background_class_id] = 0.0

                    class_id = int(np.argmax(probs))
                    final_score = float(probs[class_id])
                    if final_score < threshold:
                        continue

                    ratio = default_aspect_ratios[layer_idx][anchor_idx]
                    default_w = default_scales[layer_idx] * np.sqrt(ratio)
                    default_h = default_scales[layer_idx] / np.sqrt(ratio)
                    center_x = (gx + 0.5) / w + tx * default_w
                    center_y = (gy + 0.5) / h + ty * default_h
                    width = np.exp(tw) * default_w
                    height = np.exp(th) * default_h

                    boxes.append([
                        max(0.0, center_x - width / 2.0) * configs.IMAGE_SIZE_X,
                        max(0.0, center_y - height / 2.0) * configs.IMAGE_SIZE_Y,
                        min(1.0, center_x + width / 2.0) * configs.IMAGE_SIZE_X,
                        min(1.0, center_y + height / 2.0) * configs.IMAGE_SIZE_Y,
                    ])
                    scores.append(final_score)
                    labels.append(class_id)

    return nms(boxes, scores, labels, max_output_size=50)


def detect_rcnn(model, image_bgr, threshold=0.05):
    """Selective search proposals, classify/regress each crop, then NMS. Boxes are in resized-image pixels."""
    from common import decode_rcnn, extract_rois

    rois = [tuple(int(v) for v in r) for r in extract_rois(image_bgr)[:configs.ROIS_PER_IMAGE] if r[2] > 0 and r[3] > 0]
    if not rois:
        return [], [], []
    crops = np.asarray([
        cv2.resize(image_bgr[y:y + h, x:x + w], (configs.IMAGE_SIZE_X, configs.IMAGE_SIZE_Y), interpolation=cv2.INTER_CUBIC)
        for x, y, w, h in rois
    ], dtype=np.float32) / 255.0
    probs, deltas = [], []
    for start in range(0, len(crops), 128):
        p, d = model(tf.convert_to_tensor(crops[start:start + 128]), training=False)
        probs.append(p.numpy())
        deltas.append(d.numpy())
    return decode_rcnn(
        rois, np.concatenate(probs), np.concatenate(deltas), configs.VOC_LABELS['background'],
        configs.IMAGE_SIZE_X, configs.IMAGE_SIZE_Y, threshold=threshold,
    )


def build_model(model_name):
    if model_name == 'yolo':
        model_cls = getattr(models, configs.MODEL)
        return model_cls(
            input_shape=(configs.IMAGE_SIZE_X, configs.IMAGE_SIZE_Y, 3),
            num_classes=len(configs.VOC_LABELS),
            grid_size=configs.GRID_SIZE,
            anchors=configs.YOLO_ANCHORS,
        )
    if model_name == 'rcnn':
        return models.RCNN(
            input_shape=(configs.IMAGE_SIZE_X, configs.IMAGE_SIZE_Y, 3),
            num_class=len(configs.VOC_LABELS),
        )
    return models.SSD_Simple(
        input_shape=(None, configs.IMAGE_SIZE_X, configs.IMAGE_SIZE_Y, 3),
        num_classes=len(configs.VOC_LABELS),
    )


def load_checkpoint(model, checkpoint):
    if os.path.isdir(checkpoint):
        try:
            return tf.keras.models.load_model(checkpoint, compile=False)
        except Exception as exc:
            print(f'SavedModel load failed ({exc}); trying weight files')
            for name in ('weights.h5', 'model.weights.h5', 'ssd_model.h5'):
                candidate = os.path.join(checkpoint, name)
                if os.path.exists(candidate):
                    model.load_weights(candidate)
                    return model
            raise FileNotFoundError(f'No usable checkpoint weights found in directory: {checkpoint}')
    model.load_weights(checkpoint)
    return model


def draw_boxes(image, boxes, scores, labels):
    image = image.copy()
    id_to_name = {idx: name for name, idx in configs.VOC_LABELS.items()}
    for box, score, class_id in zip(boxes, scores, labels):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f'{id_to_name.get(class_id, f"class_{class_id}")}: {score:.2f}',
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return image


DECODERS = {'yolo': decode_yolo, 'ssd': decode_ssd}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=MODEL_CHOICES, help='Model to test')
    parser.add_argument('--image-path', type=str, required=True, help='Path to the image to test')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to a saved model checkpoint/directory')
    parser.add_argument('--threshold', type=float, default=0.2, help='Detection threshold')
    parser.add_argument('--save-path', type=str, default=None, help='Optional output image save path')
    args = parser.parse_args()

    load_modules(args.model)

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f'Image not found: {args.image_path}')

    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f'Could not read image: {args.image_path}')

    original_h, original_w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (configs.IMAGE_SIZE_X, configs.IMAGE_SIZE_Y), interpolation=cv2.INTER_CUBIC)
    input_tensor = tf.convert_to_tensor([resized.astype(np.float32) / 255.0])

    model = load_checkpoint(build_model(args.model), args.checkpoint)

    if args.model == 'rcnn':
        # trained on cv2.imread (BGR) images, so feed BGR
        resized_bgr = cv2.resize(image, (configs.IMAGE_SIZE_X, configs.IMAGE_SIZE_Y), interpolation=cv2.INTER_CUBIC)
        boxes, scores, labels = detect_rcnn(model, resized_bgr, threshold=args.threshold)
    else:
        prediction = model(input_tensor, training=False)
        boxes, scores, labels = DECODERS[args.model](prediction, model, threshold=args.threshold)

    scale_x = original_w / configs.IMAGE_SIZE_X
    scale_y = original_h / configs.IMAGE_SIZE_Y
    boxes = [[x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y] for x1, y1, x2, y2 in boxes]

    id_to_name = {idx: name for name, idx in configs.VOC_LABELS.items()}
    print(f'Detections: {len(boxes)}')
    for object_id, (box, score, class_id) in enumerate(zip(boxes, scores, labels)):
        print(
            f'object_id={object_id}, voc_class={id_to_name.get(class_id, "unknown")}({class_id}), '
            f'score={score:.3f}, bb=[{", ".join(f"{v:.1f}" for v in box)}]'
        )

    output_image = cv2.cvtColor(draw_boxes(image_rgb, boxes, scores, labels), cv2.COLOR_RGB2BGR)

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
        cv2.imwrite(args.save_path, output_image)
        print(f'Saved visualization to {args.save_path}')

    cv2.imshow(f'{args.model.upper()} detection preview', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
