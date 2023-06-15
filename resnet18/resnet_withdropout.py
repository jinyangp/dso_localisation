# Similar to resnet.py but
# 1. Added Dropout layers after each max pooling layer (p = 0.2) and after fully conncted layer just before softmax output layer (p = 0.5)

# Adapted from https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/resnet.py
import math
from tensorflow import keras
from tensorflow.keras import layers

# Initialise the weights of neural network layers
# VarianceScaling is a particular method used to initialise weights
kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

# Make 3x3 convolutional filters
def conv3x3(x, out_planes, stride=1, name=None):
    x = layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
    
    # Make 2D convolution layer
    # filters = <lefotffat>
    return layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=name)(x)

def basic_block(x, planes, stride=1, downsample=None, name=None):
    identity = x
    
    
    out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
    out = layers.ReLU(name=f'{name}.relu1')(out)

    out = conv3x3(out, planes, name=f'{name}.conv2')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)
    
    if downsample is not None:
        # Create an identical layer for each layer in the downsample
        for layer in downsample:
            identity = layer(identity)
    
    # Performs element-wise addition of multiple inputs. It is used to combine or merge the outputs of two or more layers by adding them together
    out = layers.Add(name=f'{name}.add')([identity, out])    
    out = layers.ReLU(name=f'{name}.relu2')(out)

    return out

def make_layer(x, planes, blocks, stride=1, name=None):
    downsample = None
    
    # inplanes refer to the number of channels in filters
    inplanes = x.shape[3]
    
    # Check whether we are downsampling our data (i.e. not going through every elememt)
    # This happens under two circumstances:
    # 1. when stride != 1
    # 2. when the layer we want to make has no. of channels less than the no. of channels input has
    if stride != 1 or inplanes != planes:
        # downsample consists of a Conv2D layer and a BatchNormalization layer
        downsample = [
            layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=f'{name}.0.downsample.0'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
        ]
    
    # If no downsample, downsample = None
    
    x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
    for i in range(1, blocks):
        x = basic_block(x, planes, name=f'{name}.{i}')

    return x

def resnet(x, blocks_per_layer, num_classes=1000):
    
    # ---------------------------------
    # Initial entry block
    x = layers.ZeroPadding2D(padding=3, name='conv1_pad')(x)
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal, name='conv1')(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)
    
    # Additional - Addition of a dropout layer
    x = layers.Dropout(0.2)(x)
    # ---------------------------------
    
    # ---------------------------------
    # This block of code creates the ResNet blocks
    # In ResNet-18, only have 2 layers per block
    x = make_layer(x, 64, blocks_per_layer[0], name='layer1')
    x = make_layer(x, 128, blocks_per_layer[1], stride=2, name='layer2')
    x = make_layer(x, 256, blocks_per_layer[2], stride=2, name='layer3')
    x = make_layer(x, 512, blocks_per_layer[3], stride=2, name='layer4')
    # ---------------------------------
    
    # ---------------------------------
    x = layers.GlobalAveragePooling2D(name='avgpool')(x)
    
    # Addition of a dropout layer
    x = layers.Dropout(0.2)(x)

    initializer = keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))
    x = layers.Dense(units=num_classes, kernel_initializer=initializer, bias_initializer=initializer, name='fc')(x)
    
    # Additional - Addition of a dropout layer
    x = layers.Dropout(0.5)(x)
    
    # Softmax output layer
    x = layers.Dense(units=num_classes, activation='softmax')(x)  
    # ---------------------------------

    return x

def resnet18(x, **kwargs):
    return resnet(x, [2, 2, 2, 2], **kwargs)

def resnet34(x, **kwargs):
    return resnet(x, [3, 4, 6, 3], **kwargs)
