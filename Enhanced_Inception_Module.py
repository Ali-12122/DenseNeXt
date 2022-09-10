import torch
import torch.nn as nn


class EnhancedInceptionModule(nn.Module):
    def __init__(self, input_data_depth, output_data_depth, number_of_convolution_filters=32, max_kernel_size=7,
                 dimensions_of_convolution=2):
        super().__init__()
        # The Member values are functions used in the inception, the convolution, max pool, and 1x1 convolution
        # each made with three versions to handle one, two, and three dimensional data,

        self.input_data_depth = input_data_depth
        self.output_data_depth = output_data_depth

        # Convolution layers,

        self.convolution_1d = nn.Conv1d(in_channels=number_of_convolution_filters,
                                        out_channels=number_of_convolution_filters, kernel_size=2, padding='same',
                                        bias=False)
        self.convolution_2d = nn.Conv2d(in_channels=number_of_convolution_filters,
                                        out_channels=number_of_convolution_filters, kernel_size=2, padding='same',
                                        bias=False)
        self.convolution_3d = nn.Conv3d(in_channels=number_of_convolution_filters,
                                        out_channels=number_of_convolution_filters, kernel_size=2, padding='same',
                                        bias=False)

        # 1x1 Convolution layers,

        self.convolution_1d_1x1 = nn.Conv1d(in_channels=input_data_depth,
                                            out_channels=number_of_convolution_filters, kernel_size=1, bias=False)
        self.convolution_2d_1x1 = nn.Conv2d(in_channels=input_data_depth,
                                            out_channels=number_of_convolution_filters, kernel_size=1, bias=False)
        self.convolution_3d_1x1 = nn.Conv3d(in_channels=input_data_depth,
                                            out_channels=number_of_convolution_filters, kernel_size=1, bias=False)

        # Max pooling layers,

        self.max_pool_1d = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.max_pool_2d = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.max_pool_3d = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)

        self.max_kernel_size = max_kernel_size
        self.dimensions_of_convolution = dimensions_of_convolution

    def forward(self, input_data):

        convolution = self.convolution_2d
        max_pool = self.max_pool_2d
        convolution_1x1 = self.convolution_2d_1x1

        if self.dimensions_of_convolution == 1:
            convolution = self.convolution_1d
            max_pool = self.max_pool_1d
            convolution_1x1 = self.convolution_1d_1x1

        elif self.dimensions_of_convolution == 2:
            convolution = self.convolution_2d
            max_pool = self.max_pool_2d
            convolution_1x1 = self.convolution_2d_1x1

        elif self.dimensions_of_convolution == 3:
            convolution = self.convolution_3d
            max_pool = self.max_pool_3d
            convolution_1x1 = self.convolution_3d_1x1

        # an else statement to handle invalid dimensions, if the variable dimensions_of_convolution
        # is not equal to 1 or 2 or 3.

        else:
            convolution_dimensional_error = "Invalid convolution dimensions."
            return convolution_dimensional_error

        # The Max pool branch from the inception module (future reminder: put a link to the enhanced inception here)

        convolution_input = convolution_1x1(input_data)
        collective_data = convolution_1x1(input_data)

        # Preparing the data for the iteration of convolution operation.

        max_pool_output = max_pool(input_data)
        convolution_1x1_output = convolution_1x1(max_pool_output)
        torch.cat((collective_data, convolution_1x1_output), 0)

        # Iterating the convolution operation over the data (max_kernel_size - 1) times,
        # Explanation:
        #
        # Tensor height/width after convolution = (height/width before - kernel height/width)/ stride + 1
        # more concisely H/W_out = (H/W_in - H/W_kernel)/s +1
        # at H/W_kernel = 2, s = 1
        #
        # H/W_out = ((H/W_in - 2)/ 1 + 1
        #
        # H/W_out = H/W_in - 1
        #
        # Therefore, to do a 7x7 convolution, where a 7x7 partition of the tensor is reduced
        # to a single 1x1 square, requires 6 2x2 convolutions, and a 5x5 requires 4,
        # a 3x3 requires 2.
        #
        # and a result, it is self-evident from the general pattern beforehand established,
        # that a convolution of Kernel size K, requires K-1 (2x2) convolutions.
        # ( or 1x2 or 2x2x2 for one-dimensional and three-dimensional convolutions.

        for i in range(self.max_kernel_size - 1):
            convolution_output = convolution(convolution_input)
            collective_data = torch.cat((collective_data, convolution_output), 0)
            convolution_input = convolution_output

        # Applying 1x1 convolution to change the depth of the collective_data as to match the
        # desired output depth.

        if self.dimensions_of_convolution == 1:
            output_1x1_convolution = nn.Conv1d(in_channels=collective_data.size(dim=0),
                                               out_channels=self.output_data_depth, kernel_size=1)
            return output_1x1_convolution(collective_data)

        elif self.dimensions_of_convolution == 2:
            output_1x1_convolution = nn.Conv2d(in_channels=collective_data.size(dim=0),
                                               out_channels=self.output_data_depth, kernel_size=1)
            return output_1x1_convolution(collective_data)

        else:
            output_1x1_convolution = nn.Conv3d(in_channels=collective_data.size(dim=0),
                                               out_channels=self.output_data_depth, kernel_size=1)
            return output_1x1_convolution(collective_data)
