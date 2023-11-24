
from typing import Any, Callable, List, Optional, Sequence
import numpy as np

import copy
import math
import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models as models
from torchvision.models.efficientnet import MBConvConfig, MBConv, ConvNormActivation, _efficientnet_conf
        
    
class EfficientNetEncoder(nn.Module):
    """Constructs a shufflenetv2 model with varying number of input images.
    """
    def __init__(
            self,
            inverted_residual_setting: List[MBConvConfig],
            dropout: float,
            num_input_images = 1,
            stochastic_depth_prob: float = 0.2,
            num_classes: int = 1000,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            **kwargs: Any
    ) -> None:
        """
        EfficientNet main class

        Args:
            inverted_residual_setting (List[MBConvConfig]): Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (isinstance(inverted_residual_setting, Sequence) and
                  all([isinstance(s, MBConvConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[MBConvConfig]")
        
        self.num_ch_enc = np.array([32, 24, 40, 112, 320])

        if block is None:
            block = MBConv

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvNormActivation(3 * num_input_images, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                                         activation_layer=nn.SiLU))

        # building inverted residual blocks
        total_stage_blocks = sum([cnf.num_layers for cnf in inverted_residual_setting])
        stage_block_id = 0
        for cnf in inverted_residual_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / total_stage_blocks

                stage.append(block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(nn.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 4 * lastconv_input_channels
        layers.append(ConvNormActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                         norm_layer=norm_layer, activation_layer=nn.SiLU))

        self.features = nn.Sequential(*layers)
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=dropout, inplace=True),
        #     nn.Linear(lastconv_output_channels, num_classes),
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / math.sqrt(m.out_features)
                nn.init.uniform_(m.weight, -init_range, init_range)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> List[Tensor]:
        outputs = []
        x = self.features[0](x)
        outputs.append(x)   # out [H/2, W/2, 32]

        x = self.features[1](x)
        x = self.features[2](x)
        outputs.append(x)   # out [H/4, W/4, 24]

        x = self.features[3](x)
        outputs.append(x)   # out [H/8, W/8, 40]

        x = self.features[4](x)
        x = self.features[5](x)
        outputs.append(x)   # out [H/16, W/16, 112]

        x = self.features[6](x)
        x = self.features[7](x)
        outputs.append(x)   # out [H/32, W/32, 320]

        return outputs

    def forward(self, x: Tensor) -> List[Tensor]:
        return self._forward_impl(x)

def _efficientnet_encoder(
    inverted_residual_setting: List[MBConvConfig],
    dropout: float,
    pretrained: bool,
    pretrained_path: str,
    num_input_images = 1,
    **kwargs: Any
) -> EfficientNetEncoder:
    model = EfficientNetEncoder(inverted_residual_setting, dropout, num_input_images,**kwargs)
    if pretrained:
        state_dict = torch.load(pretrained_path)
        if num_input_images > 1:
            state_dict['features.0.0.weight'] = torch.cat(
                [state_dict['features.0.0.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(state_dict, strict=False)
    return model


def efficientnet_b0(pretrained: bool = False, pretrained_path: str = "", num_input_images = 1, **kwargs: Any) -> EfficientNetEncoder:
    """
    Constructs a EfficientNet B0 architecture from
    `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks" <https://arxiv.org/abs/1905.11946>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inverted_residual_setting = _efficientnet_conf(width_mult=1.0, depth_mult=1.0, **kwargs)
    return _efficientnet_encoder(inverted_residual_setting, 0.2, pretrained, pretrained_path, num_input_images, **kwargs)