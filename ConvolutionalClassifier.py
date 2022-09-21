import torch
import torch.nn as nn


class ConvolutionalClassifier(nn.Module):
    def __init__(self, classifier_input_size, number_of_classes, activation_function=nn.ReLU(inplace=True)):
        super(ConvolutionalClassifier, self).__init__()
        self.classifier_input_size = classifier_input_size
        self.number_of_classes = number_of_classes
        self.activation_function = activation_function
        self.convolutional_classifier = nn.Sequential(
            nn.Conv1d(self.classifier_input_size, 1024, kernel_size=1), activation_function,
            nn.Conv1d(1024, 768, kernel_size=1), activation_function,
            nn.Conv1d(768, 512, kernel_size=1), activation_function,
            nn.Conv1d(512, 256, kernel_size=1), activation_function,
            nn.Conv1d(256, 128, kernel_size=1), activation_function,
            nn.Conv1d(128, 64, kernel_size=1), activation_function,
            nn.Conv1d(64, 32, kernel_size=1), activation_function,
            nn.Conv1d(32, 16, kernel_size=1), activation_function,
            nn.Conv1d(16, number_of_classes, kernel_size=1)
        )

    def forward(self, input_data):
        data_tensor = torch.flatten(input_data)
        data_tensor = torch.unsqueeze(data_tensor, 1)
        data_tensor = torch.unsqueeze(data_tensor, 0)
        data_tensor = self.convolutional_classifier(data_tensor)
        prediction_vector = torch.squeeze(data_tensor)
        return prediction_vector
