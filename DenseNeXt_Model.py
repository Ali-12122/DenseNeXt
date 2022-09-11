import math
import torch
import torch.nn as nn
import torch._tensor as tensor
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from Enhanced_Inception_Module import EnhancedInceptionModule as Inception
from collections import OrderedDict


def _batch_norm_function_factory(norm, relu, conv):
    def batch_norm_function(*inputs):
        concatenated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concatenated_features)))
        return bottleneck_output

    return batch_norm_function


class _DenseLayer(nn.Module):
    def __init__(self, number_of_input_features, growth_rate, batch_norm_size, drop_rate,
                 model_dimensions_of_convolution, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('batch_norm_1_1d', nn.BatchNorm1d(number_of_input_features)),
        self.add_module('batch_norm_1_2d', nn.BatchNorm2d(number_of_input_features)),
        self.add_module('batch_norm_1_3d', nn.BatchNorm3d(number_of_input_features)),

        self.add_module('relu1', nn.ReLU(inplace=True)),

        self.add_module('convolution_1d_1x1', nn.Conv1d(number_of_input_features, batch_norm_size * growth_rate,
                                                        kernel_size=1, stride=1, bias=False)),
        self.add_module('convolution_2d_1x1', nn.Conv2d(number_of_input_features, batch_norm_size * growth_rate,
                                                        kernel_size=1, stride=1, bias=False)),
        self.add_module('convolution_3d_1x1', nn.Conv3d(number_of_input_features, batch_norm_size * growth_rate,
                                                        kernel_size=1, stride=1, bias=False)),

        self.add_module('batch_norm_2_1d', nn.BatchNorm1d(batch_norm_size * growth_rate)),
        self.add_module('batch_norm_2_2d', nn.BatchNorm2d(batch_norm_size * growth_rate)),
        self.add_module('batch_norm_2_3d', nn.BatchNorm3d(batch_norm_size * growth_rate)),

        self.add_module('relu2', nn.ReLU(inplace=True)),

        self.add_module('inception',
                        Inception(
                            input_data_depth=batch_norm_size * growth_rate,
                            output_data_depth=growth_rate,
                            number_of_convolution_filters=16,
                            max_kernel_size=11,
                            dimensions_of_convolution=model_dimensions_of_convolution)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *previous_features):

        batch_norm_function = _batch_norm_function_factory(self.batch_norm_1_2d,
                                                           self.relu1,
                                                           self.convolution_2d_1x1)

        if self.dimensions_of_convolution == 1:
            batch_norm_function = _batch_norm_function_factory(self.batch_norm_1_1d,
                                                               self.relu1,
                                                               self.convolution_1d_1x1)
        elif self.dimensions_of_convolution == 3:
            batch_norm_function = _batch_norm_function_factory(self.batch_norm_1_3d,
                                                               self.relu1,
                                                               self.convolution_3d_1x1)

        if self.efficient and any(previous_feature.requires_grad for previous_feature in previous_features):
            bottleneck_output = cp.checkpoint(batch_norm_function, *previous_features)
        else:
            bottleneck_output = batch_norm_function(*previous_features)

        new_features = self.inception(self.relu2(self.batch_norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Module):
    def __init__(self, number_of_input_features, num_output_features, model_dimensions_of_convolution):
        super(_Transition, self).__init__()
        self.model_dimensions_of_convolution = model_dimensions_of_convolution

        self.add_module('batch_norm_1d', nn.BatchNorm1d(number_of_input_features))
        self.add_module('batch_norm_2d', nn.BatchNorm2d(number_of_input_features))
        self.add_module('batch_norm_3d', nn.BatchNorm3d(number_of_input_features))

        self.add_module('relu', nn.ReLU(inplace=True))

        self.add_module('convolution_1x1_1d', nn.Conv1d(number_of_input_features, num_output_features,
                                                        kernel_size=1, stride=1, bias=False))
        self.add_module('convolution_1x1_2d', nn.Conv2d(number_of_input_features, num_output_features,
                                                        kernel_size=1, stride=1, bias=False))
        self.add_module('convolution_1x1_3d', nn.Conv3d(number_of_input_features, num_output_features,
                                                        kernel_size=1, stride=1, bias=False))

        self.add_module('average_pool_1d', nn.AvgPool1d(kernel_size=2, stride=2))
        self.add_module('average_pool_2d', nn.AvgPool2d(kernel_size=2, stride=2))
        self.add_module('average_pool_3d', nn.AvgPool3d(kernel_size=2, stride=2))

    def forward(self, input_data):
        if self.model_dimensions_of_convolution == 1:
            return self.average_pool_1d(self.convolution_1x1_1d(self.relu(self.batch_norm_1d(input_data))))
        elif self.model_dimensions_of_convolution == 2:
            return self.average_pool_2d(self.convolution_1x1_2d(self.relu(self.batch_norm_2d(input_data))))
        elif self.model_dimensions_of_convolution == 3:
            return self.average_pool_3d(self.convolution_1x1_3d(self.relu(self.batch_norm_3d(input_data))))

        # an else statement to handle invalid dimensions, if the variable dimensions_of_convolution
        # is not equal to 1 or 2 or 3.

        else:
            convolution_dimensional_error = "Invalid convolution dimensions."
            return convolution_dimensional_error


class _DenseBlock(nn.Module):
    def __init__(self, number_of_layers, number_of_input_features, batch_norm_size, growth_rate,
                 drop_rate, model_dimensions_of_convolution, efficient=False):
        super(_DenseBlock, self).__init__()
        self.dense_layer = _DenseLayer(number_of_input_features=number_of_input_features,
                                       growth_rate=growth_rate, batch_norm_size=batch_norm_size,
                                       model_dimensions_of_convolution=model_dimensions_of_convolution,
                                       drop_rate=drop_rate, efficient=efficient
                                       )
        self.number_of_layers = number_of_layers

    def forward(self, initial_features):
        features = torch.tensor(initial_features)
        for i in range(self.number_of_layers):
            new_features = self.dense_layer(features)
            features = torch.cat((features, new_features), dim=0)
            self.dense_layer.number_of_input_features += i * self.growth_rate
        return features


class DenseNeXt(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        dense_block_configuration (list of 3 or 4 ints) - how many layers in each pooling block
        number_of_initial_features (int) - the number of filters to learn in the first convolution layer
        batch_norm_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. batch_norm_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        number_of_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """

    def __init__(self, input_data_depth=3, growth_rate=12, dense_block_configuration=(16, 16, 16), compression=0.5,
                 number_of_initial_features=24, batch_norm_size=4, drop_rate=0, number_of_classes=10,
                 model_dimensions_of_convolution=2, small_inputs=True, efficient=False, inception_convolution_filters=16
                 ):
        super(DenseNeXt, self).__init__()
        assert 0 < compression <= 1, 'compression of DenseNeXt should be between 0 and 1'
        assert 1 <= model_dimensions_of_convolution <= 3, 'model_dimensions_of_convolution should be between 1 and 3'

        if small_inputs:
            self.layers.add_module('initial_inception', Inception(input_data_depth=input_data_depth,
                                                                  output_data_depth=number_of_initial_features,
                                                                  number_of_convolution_filters=inception_convolution_filters,
                                                                  max_kernel_size=7,
                                                                  dimensions_of_convolution=model_dimensions_of_convolution
                                                                  ))
        else:
            self.layers.add_module('initial_inception', Inception(input_data_depth=input_data_depth,
                                                                  output_data_depth=number_of_initial_features,
                                                                  number_of_convolution_filters=inception_convolution_filters,
                                                                  dimensions_of_convolution=model_dimensions_of_convolution,
                                                                  max_kernel_size=7
                                                                  ))
            if model_dimensions_of_convolution == 1:
                self.layer.add_module('initial_batch_norm', nn.BatchNorm1d(number_of_initial_features))
                self.layer.add_module('initial_max_pool',
                                      nn.MaxPool1d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
            elif model_dimensions_of_convolution == 2:
                self.layer.add_module('initial_batch_norm', nn.BatchNorm2d(number_of_initial_features))
                self.layer.add_module('initial_max_pool',
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
            elif model_dimensions_of_convolution == 3:
                self.layer.add_module('initial_batch_norm', nn.BatchNorm3d(number_of_initial_features))
                self.layer.add_module('initial_max_pool',
                                      nn.MaxPool3d(kernel_size=3, stride=2, padding=1, ceil_mode=False))

            self.layer.add_module('initial_ReLU', nn.ReLU(inplace=True))

        number_of_features = number_of_initial_features
        for i, number_of_layers in enumerate(dense_block_configuration):
            block = _DenseBlock(
                number_of_layers=number_of_layers,
                number_of_input_features=number_of_features,
                batch_norm_size=batch_norm_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
                model_dimensions_of_convolution=model_dimensions_of_convolution
            )
            self.features.add_module('dense_block_%d' % (i + 1), block)

            number_of_features = number_of_features + number_of_layers * growth_rate
            if i != len(dense_block_configuration) - 1:
                transition = _Transition(number_of_input_features=number_of_features,
                                         num_output_features=int(number_of_features * compression),
                                         model_dimensions_of_convolution=model_dimensions_of_convolution
                                         )
                self.features.add_module('transition_%d' % (i + 1), transition)
                number_of_features = int(number_of_features * compression)

        if model_dimensions_of_convolution == 1:
            self.layer.add_module('final_batch_norm', nn.BatchNorm1d(number_of_initial_features))

        elif model_dimensions_of_convolution == 2:
            self.layer.add_module('final_batch_norm', nn.BatchNorm2d(number_of_initial_features))

        elif model_dimensions_of_convolution == 3:
            self.layer.add_module('final_batch_norm', nn.BatchNorm3d(number_of_initial_features))

        self.classifier = nn.Linear(number_of_features, number_of_classes)
        # TODO: try convolution as a classifier

        for name, parameter in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                n = parameter.size(0) * parameter.size(2) * parameter.size(3)
                parameter.data.normal_().mul_(math.sqrt(2. / n))
            elif 'norm' in name and 'weight' in name:
                parameter.data.fill_(1)
            elif 'norm' in name and 'bias' in name:
                parameter.data.fill_(0)
            elif 'classifier' in name and 'bias' in name:
                parameter.data.fill_(0)

    def forward(self, x):
        features = self.layers(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
