import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout, LeakyReLU
import cv2



class SSD_Simple(Model):

    def __init__(self, input_shape=(None, 300, 300, 3), num_classes=21, num_anchors_per_layer=None):
        
        super(SSD_Simple, self).__init__()
        if num_anchors_per_layer is None:
            num_anchors_per_layer = [4, 6, 6]

        self.num_classes = num_classes
        self.num_anchors_per_layer = list(num_anchors_per_layer)
        # For 300x300 input with this backbone:
        # after pool1 -> 150x150, pool2 -> 75x75, pool3 -> 37x37,
        # then stride-2 convs -> 19x19 and 10x10.
        self.feature_map_sizes = [(37, 37), (19, 19), (10, 10)]
        self.prediction_size = sum(
            h * w * anchors * (4 + self.num_classes)
            for (h, w), anchors in zip(self.feature_map_sizes, self.num_anchors_per_layer)
        )

        # 1. VGG-style backbone
        self.conv1_1 = Conv2D(64, (3, 3), padding='same', input_shape=input_shape)
        self.leaky1_1 = LeakyReLU(alpha=0.1)
        self.conv1_2 = Conv2D(64, (3, 3), padding='same')
        self.leaky1_2 = LeakyReLU(alpha=0.1)
        self.pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv2_1 = Conv2D(128, (3, 3), padding='same')
        self.leaky2_1 = LeakyReLU(alpha=0.1)
        self.conv2_2 = Conv2D(128, (3, 3), padding='same')
        self.leaky2_2 = LeakyReLU(alpha=0.1)
        self.pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv3_1 = Conv2D(256, (3, 3), padding='same')
        self.leaky3_1 = LeakyReLU(alpha=0.1)
        self.conv3_2 = Conv2D(256, (3, 3), padding='same')
        self.leaky3_2 = LeakyReLU(alpha=0.1)
        self.conv3_3 = Conv2D(256, (3, 3), padding='same')
        self.leaky3_3 = LeakyReLU(alpha=0.1)
        self.pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        self.conv4_1 = Conv2D(512, (3, 3), padding='same')
        self.leaky4_1 = LeakyReLU(alpha=0.1)
        self.conv4_2 = Conv2D(512, (3, 3), padding='same')
        self.leaky4_2 = LeakyReLU(alpha=0.1)
        self.conv4_3 = Conv2D(512, (3, 3), padding='same')
        self.leaky4_3 = LeakyReLU(alpha=0.1)

        # 2. SSD-style extra feature maps for several detection grids
        self.conv5_1 = Conv2D(256, (1, 1), padding='same')
        self.leaky5_1 = LeakyReLU(alpha=0.1)
        self.conv5_2 = Conv2D(512, (3, 3), strides=(2, 2), padding='same')
        self.leaky5_2 = LeakyReLU(alpha=0.1)

        self.conv6_1 = Conv2D(128, (1, 1), padding='same')
        self.leaky6_1 = LeakyReLU(alpha=0.1)
        self.conv6_2 = Conv2D(256, (3, 3), strides=(2, 2), padding='same')
        self.leaky6_2 = LeakyReLU(alpha=0.1)

        # 3. One detection head per feature map
        self.pred_heads = [
            Conv2D(self.num_anchors_per_layer[i] * (4 + self.num_classes), (3, 3), padding='same')
            for i in range(len(self.num_anchors_per_layer))
        ]

    def call(self, inputs):
        x = self.conv1_1(inputs)
        x = self.leaky1_1(x)
        x = self.conv1_2(x)
        x = self.leaky1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.leaky2_1(x)
        x = self.conv2_2(x)
        x = self.leaky2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.leaky3_1(x)
        x = self.conv3_2(x)
        x = self.leaky3_2(x)
        x = self.conv3_3(x)
        x = self.leaky3_3(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.leaky4_1(x)
        x = self.conv4_2(x)
        x = self.leaky4_2(x)
        x = self.conv4_3(x)
        x = self.leaky4_3(x)
        feat1 = x
        #print("Feature map 1 shape:", feat1.shape)

        x = self.conv5_1(x)
        x = self.leaky5_1(x)
        x = self.conv5_2(x)
        x = self.leaky5_2(x)
        feat2 = x
        #print("Feature map 2 shape:", feat2.shape)

        x = self.conv6_1(x)
        x = self.leaky6_1(x)
        x = self.conv6_2(x)
        x = self.leaky6_2(x)
        feat3 = x
        #print("Feature map 3 shape:", feat3.shape)

        # the final prediction heads for detection per feature map
        # each feature map predicts #anchors * (4 + num_classes) prediction 
        # per item in the feature map
        feature_maps = [feat1, feat2, feat3]
        outputs = [self.pred_heads[i](feature_maps[i]) for i in range(len(feature_maps))]
        #print("Output shapes:", [output.shape for output in outputs])
        return outputs

