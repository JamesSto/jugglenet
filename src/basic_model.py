import torch
import torch.nn as nn

LAYER_SIZE = 512

class BasicNetwork(nn.Module):
    def __init__(self, image_shape, output_size):
        super(BasicNetwork, self).__init__()

        self.input_size = image_shape[0] * image_shape[1]
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
