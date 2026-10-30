DATA_PATH = '/home/kishky/master/git/vision/VOC2012'

# ssd-style feature extractor requires a 300x300
IMAGE_SIZE_X = 300
IMAGE_SIZE_Y = 300

BATCH_SIZE = 8
LEARNING_RATE = 1e-04
TRAIN_FILE = 'train.txt'
VAL_FILE = 'val.txt'

VOC_LABELS = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3,
                'bottle': 4, 'bus': 5, 'car': 6, 'cat': 7, 'chair': 8,
                'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12,
                'motorbike': 13, 'person': 14, 'pottedplant': 15,
                'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19, 'background': 20}

LOG_EVERY_STEP = 100
SAVE_EVERY_EPOCH = 10

MODEL = 'SSD_Simple'