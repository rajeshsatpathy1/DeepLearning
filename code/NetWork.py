import torch
from torch.functional import Tensor
import torch.nn as nn
from torch.nn.modules.conv import Conv2d

""" This script defines the network.
"""

class ResNet(nn.Module):
    def __init__(self,
            resnet_size,
            num_classes,
            first_num_filters,
        ):
        """
        Args:
            resnet_size: A positive integer (n).
            num_classes: A positive integer. Define the number of classes.
            first_num_filters: An integer. The number of filters to use for the
                first block layer of the model. This number is then doubled
                for each subsampling block layer.

        Architecture:
        layer_name      | start | stack1 | stack2 | stack3 | output      |
        output_map_size | 32x32 | 32X32  | 16x16  | 8x8    | 1x1         |
        #layers         | 1     | 3n     | 3n     | 3n     | 1           |
        #filters        | 128   | 128    | 256    | 512    | num_classes |

        Args:
            inputs: A Tensor representing a batch of input images.
        
        Returns:
            A logits Tensor of shape [<batch_size>, self.num_classes].
        """
        super(ResNet, self).__init__()
        self.resnet_size = resnet_size
        self.num_classes = num_classes
        self.first_num_filters = first_num_filters

        # First convolution to set to first_num_filters
        self.start_layer = nn.Conv2d(3, self.first_num_filters, 3, stride=1, padding=1, bias=False, padding_mode='zeros')

        self.batch_norm_relu_start = batch_norm_relu_layer(
            num_features=self.first_num_filters, 
            eps=1e-5, 
            momentum=0.997,
        )
        block_fn = bottleneck_block

        self.stack_layers = nn.ModuleList()
        
        for i in range(3):
            filters = self.first_num_filters * (2**i)
            strides = 1 if i == 0 else 2
            self.stack_layers.append(stack_layer(filters, block_fn, strides, self.resnet_size, self.first_num_filters))
        self.output_layer = output_layer(filters, self.num_classes)
    
    def forward(self, inputs):
        outputs = self.start_layer(inputs)
        outputs = self.batch_norm_relu_start(outputs)
        
        for i in range(3):
            outputs = self.stack_layers[i](outputs)
        
        outputs = self.output_layer(outputs)
        return outputs

#############################################################################
# Blocks building the network
#############################################################################

class batch_norm_relu_layer(nn.Module):
    """ Perform batch normalization then relu.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.997) -> None:
        super(batch_norm_relu_layer, self).__init__()
        
        self.BN = nn.BatchNorm2d(num_features, eps, momentum)
        self.relu = nn.ReLU(inplace=True)

        
    def forward(self, inputs: Tensor) -> Tensor:
        
        out = self.BN(inputs)
        out = self.relu(out)

        return out
        


class bottleneck_block(nn.Module):
    """ Creates a bottleneck residual block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(bottleneck_block, self).__init__()

        # Pre-activation
        self.projection_shortcut = projection_shortcut
        self.BNSB = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        
        self.convSB1 = nn.Conv2d(filters, filters//4, 1, 1, bias=False)
        self.BNSB1 = nn.BatchNorm2d(filters//4)
        self.reluSB1 = nn.ReLU(inplace=True)
        
        self.convBot = Conv2d(filters//4, filters//4, 3, 1,1, bias=False)
        self.BNBot = nn.BatchNorm2d(filters//4)
        self.reluBot = nn.ReLU(inplace=True)
        
        self.convSB2 = nn.Conv2d(filters//4, filters, 1, 1, bias=False)

        

    def forward(self, inputs: Tensor) -> Tensor:
        
        shortcut = inputs
        if(self.projection_shortcut is not None):
            # print("Inside projection")
            shortcut = self.projection_shortcut(inputs)
            shortcut = self.BNSB(shortcut)
            shortcut = self.relu(shortcut)


        out = self.convSB1(shortcut)
        out = self.BNSB1(out)
        out = self.reluSB1(out)

        out = self.convBot(out)
        out = self.BNBot(out)
        out = self.reluBot(out)

        out = self.convSB2(out)
        
        out = out + shortcut

        return out
        
class stack_layer(nn.Module):
    """ Creates one stack of bottleneck blocks.

    Args:
        filters: A positive integer. The number of filters for the first
			    convolution in a block.
		block_fn: 'bottleneck_block'
		strides: A positive integer. The stride to use for the first block. If
				greater than 1, this layer will ultimately downsample the input.
        resnet_size: #residual_blocks in each stack layer
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, block_fn, strides, resnet_size, first_num_filters) -> None:
        super(stack_layer, self).__init__()
        
        # projection_shortcut = ?
        # Only the first block per stack_layer uses projection_shortcut and strides
        layers = []
        projection_shortcut = nn.Conv2d(filters//2, filters, kernel_size=1, stride=strides, bias=False)
        
        
        if(first_num_filters == filters):
            layers.append(block_fn(filters, None, 1, first_num_filters))
        else:
            layers.append(block_fn(filters, projection_shortcut, 1, first_num_filters))
        
        for i in range(resnet_size - 1):
          layers.append(block_fn(filters, None, 1, first_num_filters))

        self.stack = nn.Sequential(*layers)
        
    
    def forward(self, inputs: Tensor) -> Tensor:
        
        out = self.stack(inputs)
        return out
        
        

class output_layer(nn.Module):
    """ Implement the output layer.

    Args:
        filters: A positive integer. The number of filters.
        num_classes: A positive integer. Define the number of classes.
    """
    def __init__(self, filters, num_classes) -> None:
        super(output_layer, self).__init__()

        self.avgPool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(filters, num_classes, bias=True)
        
    
    def forward(self, inputs: Tensor) -> Tensor:
        out = self.avgPool(inputs)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc1(out)

        return out