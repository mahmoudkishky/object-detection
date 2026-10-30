import cv2

def _to_xyxy(box, box_format):
    if box_format == 'xyxy':
        return box
    if box_format == 'cxcywh':
        cx, cy, w, h = box
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    raise ValueError(f"Unsupported box_format: {box_format}")


def compute_iou(bb1, bb2, box_format='xyxy'):
    """IoU of two boxes sharing one format: 'xyxy' = [x_min, y_min, x_max, y_max] (SSD),
    'cxcywh' = [x_center, y_center, width, height] (YOLO targets)."""
    bb1 = _to_xyxy(bb1, box_format)
    bb2 = _to_xyxy(bb2, box_format)
    x_min = max(bb1[0], bb2[0])
    y_min = max(bb1[1], bb2[1])
    x_max = min(bb1[2], bb2[2])
    y_max = min(bb1[3], bb2[3])

    if x_min > x_max or y_min > y_max:
        return 0.0
    
    intersection = (x_max - x_min) * (y_max - y_min)
    area1 = max(0.0, bb1[2] - bb1[0]) * max(0.0, bb1[3] - bb1[1])
    area2 = max(0.0, bb2[2] - bb2[0]) * max(0.0, bb2[3] - bb2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0

def extract_rois(img):

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    rois = ss.process()

    # rois are in (x, y, w, h) format, to be converted to (x_min, y_min, x_max, y_max)
    return rois

def compute_average_precision(recalls, precisions):
    """Area under the monotonic precision envelope of a precision/recall curve."""
    recall_points = [0.0] + list(recalls) + [1.0]
    precision_points = [0.0] + list(precisions) + [0.0]
    for i in range(len(precision_points) - 2, -1, -1):
        precision_points[i] = max(precision_points[i], precision_points[i + 1])
    return sum(
        (recall_points[i + 1] - recall_points[i]) * precision_points[i + 1]
        for i in range(len(recall_points) - 1)
    )


def compute_map(detections, ground_truths, num_classes, iou_threshold=0.5):
    """Mean average precision over classes.

    detections: per image, a (boxes, scores, labels) tuple, boxes as [x_min, y_min, x_max, y_max].
    ground_truths: per image, a (boxes, labels) tuple in the same coordinate space.
    Classes without ground truth are skipped. Returns (mAP, {class_id: AP}).
    """
    per_class_ap = {}
    for class_id in range(num_classes):
        gt_boxes = {
            image_id: [box for box, label in zip(boxes, labels) if label == class_id]
            for image_id, (boxes, labels) in enumerate(ground_truths)
        }
        num_gt = sum(len(boxes) for boxes in gt_boxes.values())
        if num_gt == 0:
            continue

        class_detections = [
            (score, image_id, box)
            for image_id, (boxes, scores, labels) in enumerate(detections)
            for box, score, label in zip(boxes, scores, labels)
            if label == class_id
        ]
        class_detections.sort(key=lambda item: item[0], reverse=True)

        matched = {image_id: set() for image_id in gt_boxes}
        true_positives = 0
        recalls, precisions = [], []
        for rank, (_, image_id, box) in enumerate(class_detections, start=1):
            candidates = gt_boxes.get(image_id, [])
            ious = [compute_iou(box, gt) for gt in candidates]
            best = max(range(len(ious)), key=ious.__getitem__) if ious else -1
            if best >= 0 and ious[best] >= iou_threshold and best not in matched[image_id]:
                matched[image_id].add(best)
                true_positives += 1
            recalls.append(true_positives / num_gt)
            precisions.append(true_positives / rank)

        per_class_ap[class_id] = compute_average_precision(recalls, precisions)

    mean_ap = sum(per_class_ap.values()) / len(per_class_ap) if per_class_ap else 0.0
    return mean_ap, per_class_ap


def extract_rois(img):
    """Selective search proposals for a uint8 image, as (x, y, w, h) rows."""
    import cv2

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    return ss.process()


def encode_rcnn_box(roi, gt_box):
    """R-CNN regression target (tx, ty, tw, th) of an xyxy ground truth w.r.t. an (x, y, w, h) ROI."""
    import numpy as np

    x, y, w, h = roi
    xg_min, yg_min, xg_max, yg_max = gt_box
    wg = max(xg_max - xg_min, 1e-6)
    hg = max(yg_max - yg_min, 1e-6)
    xg = xg_min + wg / 2
    yg = yg_min + hg / 2
    return [(xg - (x + w / 2)) / w, (yg - (y + h / 2)) / h, np.log(wg / w), np.log(hg / h)]


def decode_rcnn(rois, class_probs, box_deltas, background_id, image_w, image_h,
                threshold=0.05, iou_threshold=0.5, max_output_size=50):
    """Turn per-ROI classifier/regressor outputs into NMS-filtered xyxy boxes, scores and labels."""
    import numpy as np
    import tensorflow as tf

    rois = np.asarray(rois, dtype=np.float32).reshape(-1, 4)
    probs = np.array(class_probs, dtype=np.float32)
    deltas = np.asarray(box_deltas, dtype=np.float32)
    probs[:, background_id] = 0.0
    labels = np.argmax(probs, axis=-1)
    scores = np.max(probs, axis=-1)

    keep = scores >= threshold
    if not np.any(keep):
        return [], [], []

    x, y, w, h = [rois[keep][:, i] for i in range(4)]
    tx, ty, tw, th = [deltas[keep][:, i] for i in range(4)]
    cx, cy = tx * w + x + w / 2, ty * h + y + h / 2
    bw, bh = np.exp(np.clip(tw, -10, 10)) * w, np.exp(np.clip(th, -10, 10)) * h
    boxes = np.stack([
        np.clip(cx - bw / 2, 0, image_w),
        np.clip(cy - bh / 2, 0, image_h),
        np.clip(cx + bw / 2, 0, image_w),
        np.clip(cy + bh / 2, 0, image_h),
    ], axis=-1).astype(np.float32)
    scores, labels = scores[keep], labels[keep]

    selected = tf.image.non_max_suppression(boxes, scores, max_output_size, iou_threshold).numpy()
    return boxes[selected].tolist(), scores[selected].tolist(), labels[selected].tolist()
