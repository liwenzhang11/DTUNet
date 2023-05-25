from keras import layers
from keras.initializers import random_normal


def Vgg19(img_input):
    # Block 1
    # 512,512,3 -> 512,512,64
    # Block 1
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same',dilation_rate=(1,1), name='block1_conv1')(
        img_input)
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same',dilation_rate=(2,2), name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same',dilation_rate=(5,1), name='block2_conv1')(x)
    x = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same',dilation_rate=(2,5), name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same',dilation_rate=(1,1), name='block3_conv1')(x)
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same', dilation_rate=(2,2),name='block3_conv2')(x)
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same',dilation_rate=(5,1), name='block3_conv3')(x)
    x = layers.Conv2D(
        256, (3, 3), activation='relu', padding='same',dilation_rate=(2,5), name='block3_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.Conv2D(
        512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)