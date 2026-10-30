DATA_PATH = '/home/kishky/master/git/vision/VOC2012'

IMAGE_SIZE_X = 227
IMAGE_SIZE_Y = 227

BATCH_SIZE = 2
LEARNING_RATE = 5e-4

TRAIN_FILE = 'train.txt'
VAL_FILE = 'val.txt'

VOC_LABELS = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3,
                'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8,
                'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12,
                'motorbike': 13, 'person': 14, 'pottedplant': 15,
                'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19, 'background': 20}

ROIS_PER_IMAGE = 2000

MODEL = 'RCNN'

LOG_EVERY_STEP = 1
SAVE_EVERY_EPOCH = 20
