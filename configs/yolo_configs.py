DATA_PATH = '/home/kishky/master/git/vision/VOC2012'

# YOLOv1-style feature extractor requires a 448x448 input to reach the final 7x7 grid.
IMAGE_SIZE_X = 448
IMAGE_SIZE_Y = 448

BATCH_SIZE = 8
LEARNING_RATE = 5e-04

TRAIN_FILE = 'train.txt'
VAL_FILE = 'val.txt'

GRID_SIZE = (7, 7)

YOLO_ANCHORS = [
    (1.3221, 1.73145),
    (3.19275, 4.00944),
    (5.05587, 8.09892),
    (9.47112, 4.84053),
    (11.2364, 10.0071),
]


VOC_LABELS = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3,
                'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8,
                'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12,
                'motorbike': 13, 'person': 14, 'pottedplant': 15,
                'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

MODEL = 'YOLO_Simple'

LOG_EVERY_STEP = 100
SAVE_EVERY_EPOCH = 10
