import torch
import torch.nn as nn

LAYER_SIZE = 256

class BasicDenseNetwork(nn.Module):
    def __init__(self, image_shape, output_size):
        super(BasicDenseNetwork, self).__init__()
        # Assume that shape is (channels, width, height)
        self.input_size = image_shape[-1] * image_shape[-2]
        self.activation = nn.ReLU()
        self.layer1 = nn.Linear(self.input_size, LAYER_SIZE)
        self.layer2 = nn.Linear(LAYER_SIZE, LAYER_SIZE)
        self.output_layer = nn.Linear(LAYER_SIZE, output_size)
        self.softmax = nn.Softmax()

    def forward(self, image):
        # Flatten the image
        image = image.view(image.size()[0], -1)
        post_l1 = self.activation(self.layer1(image))
        post_l2 = self.activation(self.layer2(post_l1))
        return self.softmax(self.output_layer(post_l2))

class ConvolutionalNetwork(nn.Module):
    def __init__(self, image_shape, output_size):
        super(ConvolutionalNetwork, self).__init__()

        self.input_shape = image_shape

        self.activation = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=self.input_shape[0],
                               out_channels=8, 
                               kernel_size=5,
                               padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8,
                               out_channels=16, 
                               kernel_size=5,
                               padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.layer1 = nn.Linear(16 * 44 * 27, LAYER_SIZE)
        self.layer2 = nn.Linear(LAYER_SIZE, output_size)

        self.softmax = nn.Softmax()

    def forward(self, image):
        image = self.conv1(image)
        image = self.activation(image)
        image = self.pool1(image)
        image = self.activation(self.conv2(image))
        image = self.pool2(image)
        flat = image.view(image.size()[0], -1)

        layer1_out = self.activation(self.layer1(flat))
        out = self.softmax(self.layer2(layer1_out))
        return out

class LSTMNetwork(nn.Module):
    def __init__(self, output_size):
        pass