import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, LeakyReLU
import cv2

class YOLO(Model):

    def __init__(self, input_shape=(None, 448, 448, 3), num_classes=20, grid_size=(7, 7), anchors=None):

        super(YOLO, self).__init__()
        # use configuration in original YOLO paper https://arxiv.org/abs/1506.02640
        # this architecture downsamples the image 64x before the final 7x7 grid, so
        # the input must be at least 448x448; smaller sizes collapse the feature map.
        if len(input_shape) >= 3 and input_shape[1] < 448 and input_shape[2] < 448:
            raise ValueError(
                f"YOLO requires an input size of at least 448x448; got {input_shape}. "
                "Use 448x448 or larger to avoid zero-sized feature maps."
            )

        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.anchors = anchors
        self.prediction_size = self.grid_size[0] * self.grid_size[1] * self.num_anchors * (5 + self.num_classes)

        self.conv1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', input_shape=input_shape)
        self.leaky1 = LeakyReLU(alpha=0.1)
        self.pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv2 = Conv2D(filters=192, kernel_size=(3, 3), padding='same')
        self.leaky2 = LeakyReLU(alpha=0.1)
        self.pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv3 = Conv2D(filters=128, kernel_size=(1, 1), padding='same')
        self.leaky3 = LeakyReLU(alpha=0.1)
        self.conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.leaky4 = LeakyReLU(alpha=0.1)
        self.conv5 = Conv2D(filters=256, kernel_size=(1, 1), padding='same')
        self.leaky5 = LeakyReLU(alpha=0.1)
        self.conv6 = Conv2D(filters=512, kernel_size=(3, 3), padding='same')
        self.leaky6 = LeakyReLU(alpha=0.1)
        self.pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv7 = Conv2D(filters=512, kernel_size=(1, 1), padding='same')
        self.leaky7 = LeakyReLU(alpha=0.1)
        self.conv8 = Conv2D(filters=1024, kernel_size=(3, 3), padding='same')
        self.leaky8 = LeakyReLU(alpha=0.1)
        self.pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv9 = Conv2D(filters=1024, kernel_size=(3, 3), padding='same')
        self.leaky9 = LeakyReLU(alpha=0.1)
        self.conv10 = Conv2D(filters=1024, kernel_size=(3, 3), padding='same')
        self.leaky10 = LeakyReLU(alpha=0.1)
        self.conv11 = Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding='same')
        self.leaky11 = LeakyReLU(alpha=0.1)
        self.conv12 = Conv2D(filters=1024, kernel_size=(3, 3), padding='same')
        self.leaky12 = LeakyReLU(alpha=0.1)
        self.conv13 = Conv2D(filters=1024, kernel_size=(3, 3), padding='same')
        self.leaky13 = LeakyReLU(alpha=0.1)

        self.flatten = Flatten()

        self.fc1 = Dense(4096)
        self.leaky_fc1 = LeakyReLU(alpha=0.1)
        self.dropout = Dropout(0.5)
        self.fc2 = Dense(self.prediction_size, activation='linear')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.leaky1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.leaky2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.leaky3(x)
        x = self.conv4(x)
        x = self.leaky4(x)
        x = self.conv5(x)
        x = self.leaky5(x)
        x = self.conv6(x)
        x = self.leaky6(x)
        x = self.pool3(x)
        x = self.conv7(x)
        x = self.leaky7(x)
        x = self.conv8(x)
        x = self.leaky8(x)
        x = self.pool4(x)
        x = self.conv9(x)
        x = self.leaky9(x)
        x = self.conv10(x)
        x = self.leaky10(x)
        x = self.conv11(x)
        x = self.leaky11(x)
        x = self.conv12(x)
        x = self.leaky12(x)
        x = self.conv13(x)
        x = self.leaky13(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.leaky_fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class YOLO_Simple(Model):

    def __init__(self, input_shape=(None, 448, 448, 3), num_classes=20, grid_size=(7, 7), anchors=None):

        super(YOLO_Simple, self).__init__()

        if len(input_shape) >= 3 and input_shape[1] < 448 and input_shape[2] < 448:
            raise ValueError(
                f"YOLO requires an input size of at least 448x448; got {input_shape}. "
                "Use 448x448 or larger to avoid zero-sized feature maps."
            )

        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.anchors = anchors
        self.prediction_size = self.grid_size[0] * self.grid_size[1] * self.num_anchors * (5 + self.num_classes)

        self.conv1 = Conv2D(filters=32, kernel_size=(7, 7), strides=(2, 2), padding='same', input_shape=input_shape)
        self.leaky1 = LeakyReLU(alpha=0.1)
        self.pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.leaky2 = LeakyReLU(alpha=0.1)
        self.pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')
        self.leaky3 = LeakyReLU(alpha=0.1)
        self.conv4 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.leaky4 = LeakyReLU(alpha=0.1)
        self.conv5 = Conv2D(filters=128, kernel_size=(3, 3), padding='same')
        self.leaky5 = LeakyReLU(alpha=0.1)
        self.pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv6 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.leaky6 = LeakyReLU(alpha=0.1)
        self.conv7 = Conv2D(filters=256, kernel_size=(3, 3), padding='same')
        self.leaky7 = LeakyReLU(alpha=0.1)
        self.pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.flatten = Flatten()
        self.fc1 = Dense(512)
        self.leaky_fc1 = LeakyReLU(alpha=0.1)
        self.dropout = Dropout(0.2)
        self.fc2 = Dense(self.prediction_size, activation='linear')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.leaky1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.leaky2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.leaky3(x)
        x = self.conv4(x)
        x = self.leaky4(x)
        x = self.conv5(x)
        x = self.leaky5(x)
        x = self.pool3(x)

        x = self.conv6(x)
        x = self.leaky6(x)
        x = self.conv7(x)
        x = self.leaky7(x)
        x = self.pool4(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.leaky_fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x