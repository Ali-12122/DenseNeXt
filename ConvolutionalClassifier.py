import torch
import torch.nn as nn


class ConvolutionalClassifier(nn.Module):
    def __init__(self, input_data_dimensions, number_of_classes, number_of_dimensions_of_convolution,
                 number_of_convolution_filters=16, classifier_kernel_size=17):
        super(ConvolutionalClassifier, self).__init__()

        self.input_data_dimensions = input_data_dimensions
        self.number_of_classes = number_of_classes
        self.classifier_kernel_size = classifier_kernel_size
        self.number_of_dimensions_of_convolution = number_of_dimensions_of_convolution
        self.number_of_convolution_filters = number_of_convolution_filters

        if self.number_of_dimensions_of_convolution == 1:
            self.convolution_1x1 = nn.Conv1d(in_channels=input_data_dimensions.size(dim=1),
                                             out_channels=self.number_of_convolution_filters,
                                             kernel_size=1, padding='same', bias=False)
        elif self.number_of_dimensions_of_convolution == 2:
            self.convolution_1x1 = nn.Conv2d(in_channels=input_data_dimensions.size(dim=1),
                                             out_channels=self.number_of_convolution_filters,
                                             kernel_size=1, padding='same', bias=False)
        else:
            self.convolution_1x1 = nn.Conv3d(in_channels=input_data_dimensions.size(dim=1),
                                             out_channels=self.number_of_convolution_filters,
                                             kernel_size=1, padding='same', bias=False)

        for i in range(self.classifier_kernel_size):
            if self.number_of_dimensions_of_convolution == 1:
                convolution = nn.Conv1d(in_channels=number_of_convolution_filters,
                                        out_channels=number_of_convolution_filters, kernel_size=1, padding='same',
                                        bias=False)
                batch_norm = nn.BatchNorm1d(number_of_convolution_filters)
            elif self.number_of_dimensions_of_convolution == 2:
                convolution = nn.Conv2d(in_channels=number_of_convolution_filters,
                                        out_channels=number_of_convolution_filters, kernel_size=1, padding='same',
                                        bias=False)
                batch_norm = nn.BatchNorm2d(number_of_convolution_filters)
            else:
                convolution = nn.Conv3d(in_channels=number_of_convolution_filters,
                                        out_channels=number_of_convolution_filters, kernel_size=1, padding='same',
                                        bias=False)
                batch_norm = nn.BatchNorm3d(number_of_convolution_filters)
            self.add_module('convolution_1x1_%d' % (i + 1), convolution)
            self.add_module('batch_norm_%d' % (i + 1), batch_norm)

        self.relu = nn.ReLU(inplace=True)

        data_tensor_flat_length = self.input_data_dimensions(0) * self.number_of_convolution_filters
        for i in range(2, len(input_data_dimensions)):
            data_tensor_flat_length *= input_data_dimensions(i)
        self.final_1d_convolution = nn.Conv1d(data_tensor_flat_length, self.number_of_classes, kernel_size=1)

    def forward(self, input_data):
        data_tensor = self.convolution_1x1(input_data)
        for name, layer in self.named_children():
            data_tensor = layer(data_tensor)

        data_tensor = torch.flatten(input_data)
        data_tensor = torch.unsqueeze(data_tensor, 1)
        data_tensor = torch.unsqueeze(data_tensor, 0)
        data_tensor = self.final_1d_convolution(data_tensor)
        prediction_vector = torch.squeeze(data_tensor)
        return prediction_vector
