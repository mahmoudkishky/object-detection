import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


class RCNN(Model):

    def __init__(self, input_shape=(227, 227, 3), num_class=21):

        super(RCNN, self).__init__()
        self.input_shape_ = input_shape
        self.num_class = num_class

        self.conv1 = Conv2D(96, kernel_size=11, strides=4, activation='relu')
        self.pool1 = MaxPooling2D(pool_size=3, strides=2)

        self.conv2 = Conv2D(256, kernel_size=5, padding='same', activation='relu')
        self.pool2 = MaxPooling2D(pool_size=3, strides=2)

        self.conv3 = Conv2D(384, kernel_size=3, padding='same', activation='relu')
        self.conv4 = Conv2D(384, kernel_size=3, padding='same', activation='relu')
        self.conv5 = Conv2D(256, kernel_size=3, padding='same', activation='relu')
        self.pool3 = MaxPooling2D(pool_size=3, strides=2)

        self.flatten = Flatten()

        self.fc1 = Dense(4096, activation='relu')
        self.drop1 = Dropout(0.5)

        self.fc2 = Dense(4096, activation='relu')
        self.drop2 = Dropout(0.5)

        self.classifier = Dense(self.num_class, activation='softmax')
        self.regressor = Dense(4, activation='linear', name='regression')


    def call(self, inputs, training=False):
        
        x = self.conv1(inputs)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.drop1(x, training=training)
        x = self.fc2(x)
        x = self.drop2(x, training=training)

        scores = self.classifier(x)
        bboxes = self.regressor(x)
    
        return scores, bboxes



    
