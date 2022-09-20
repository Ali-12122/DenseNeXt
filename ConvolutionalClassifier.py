import torch.nn as nn


class ConvolutionalClassifier(nn.Module):
    def __init__(self, classifier_input_channels, number_of_classes,
                 model_dimensions_of_convolution, activation_function=nn.ReLU(inplace=True)):
        super(ConvolutionalClassifier, self).__init__()

        self.classifier_input_channels = classifier_input_channels
        self.number_of_classes = number_of_classes
        self.model_dimensions_of_convolution = model_dimensions_of_convolution
        self.activation_function = activation_function

        assert 1 <= model_dimensions_of_convolution <= 3, 'model_dimensions_of_convolution should be between 1 and 3'

        self.convolutional_classifier_1d = nn.Sequential(
            nn.Conv1d(self.classifier_input_channels, 1024, kernel_size=1), activation_function,
            nn.Conv1d(1024, 512, kernel_size=1), activation_function,
            nn.Conv1d(512, 256, kernel_size=1), activation_function,
            nn.Conv1d(256, 128, kernel_size=1), activation_function,
            nn.Conv1d(128, number_of_classes, kernel_size=1)
        )
        self.convolutional_classifier_2d = nn.Sequential(
            nn.Conv2d(self.classifier_input_channels, 1024, kernel_size=1), activation_function,
            nn.Conv2d(1024, 512, kernel_size=1), activation_function,
            nn.Conv2d(512, 256, kernel_size=1), activation_function,
            nn.Conv2d(256, 128, kernel_size=1), activation_function,
            nn.Conv2d(128, number_of_classes, kernel_size=1)
        )
        self.convolutional_classifier_3d = nn.Sequential(
            nn.Conv3d(self.classifier_input_channels, 1024, kernel_size=1), activation_function,
            nn.Conv3d(1024, 512, kernel_size=1), activation_function,
            nn.Conv3d(512, 256, kernel_size=1), activation_function,
            nn.Conv3d(256, 128, kernel_size=1), activation_function,
            nn.Conv3d(128, number_of_classes, kernel_size=1)
        )

    def forward(self, input_data):
        if self.model_dimensions_of_convolution == 1:
            return self.convolutional_classifier_1d(input_data)
        elif self.model_dimensions_of_convolution == 2:
            return self.convolutional_classifier_2d(input_data)
        else:
            return self.convolutional_classifier_3d(input_data)
