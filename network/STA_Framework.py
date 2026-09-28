from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch import nn
from .branch import IOD_Branch
from .dpdf import DPDFAttention
from .resnet import IOD_ResNet
from .threed_models.i3d_resnet import IOD_I3D_ResNet
from .threed_models.s3d_resnet import IOD_S3D_ResNet
from .twod_models.TAM_resnet import IOD_TAM_ResNet
from .twod_models.TEA_resnet import IOD_TEA_Res2Net
from .twod_models.TDN_resnet import IOD_TDN_ResNet
from .twod_models.TIN_resnet import IOD_TIN_ResNet
from .twod_models.TSM_resnet import IOD_TSM_ResNet
from .twod_models.MS_resnet import IOD_MS_ResNet

#you can comment TINresnet and MSresnet below for fast implementation
backbone = {
    'resnet': IOD_ResNet,
    'I3Dresnet':IOD_I3D_ResNet,
    'S3Dresnet':IOD_S3D_ResNet,
    'TAMresnet':IOD_TAM_ResNet,
    'TEAresnet':IOD_TEA_Res2Net,
    'TDNresnet':IOD_TDN_ResNet,
    'TSMresnet':IOD_TSM_ResNet,
    'TINresnet':IOD_TIN_ResNet,
    'MSresnet':IOD_MS_ResNet
    }

class STA_Framework(nn.Module):
    def __init__(self, arch, num_layers, branch_info, head_conv, K,
                 use_dpdf=False, dpdf_heads=4, dpdf_branch_channels=16,
                 dpdf_deform_kernel=5,
                 dpdf_dilation_rates=(1, 6, 12, 18),
                 dpdf_temporal=False,
                 dpdf_temporal_align=False,
                 dpdf_dynamic_dilation=False,
                 dpdf_consistency=False):
        super(STA_Framework, self).__init__()
        self.K = K
        self.backbone = backbone[arch](num_layers,K)
        self.arch = arch
        self.use_dpdf = use_dpdf
        self.dpdf_consistency = dpdf_consistency
        if self.use_dpdf:
            self.dpdf = DPDFAttention(
                channels=self.backbone.output_channel,
                heads=dpdf_heads,
                branch_channels=dpdf_branch_channels,
                deform_kernel=dpdf_deform_kernel,
                dilation_rates=dpdf_dilation_rates,
                temporal=dpdf_temporal,
                temporal_align=dpdf_temporal_align,
                dynamic_dilation=dpdf_dynamic_dilation,
            )
        self.branch = IOD_Branch(self.backbone.output_channel, arch, head_conv, branch_info, K)
        self.R2D  = nn.Sequential(
            nn.Conv2d(K * self.backbone.output_channel, self.backbone.output_channel,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True))

    def _refine_chunk(self, chunk):
        if not self.use_dpdf:
            return chunk, None
        return_aux = self.training and self.dpdf_consistency
        if self.dpdf.temporal or return_aux:
            result = self.dpdf.forward_sequence(
                chunk,
                return_aux=return_aux,
            )
            if return_aux:
                return result
            return result, None
        # One shared DPDF is applied independently to each frame. The
        # temporal backbones have already mixed information across frames;
        # keeping this refinement 2-D preserves the branch's per-frame input
        # contract for wh and STA_offset.
        return [self.dpdf(feature) for feature in chunk], None

    def _forward_branch(self, chunk):
        chunk, dpdf_aux = self._refine_chunk(chunk)
        output = self.branch(chunk)
        if dpdf_aux is not None:
            output['dpdf_aux'] = dpdf_aux
        return [output]

    def forward(self, input):
        if self.arch == 'I3Dresnet' or self.arch ==  'S3Dresnet' or self.arch =='TAMresnet' or self.arch ==  'MSresnet' or \
                self.arch == 'TEAresnet' or self.arch == 'TINresnet' or self.arch ==  'TSMresnet':
            inputlist = [input[i].unsqueeze(2) for i in range(self.K)]
            input_cat = torch.cat(inputlist, dim=2)# B C T H W
            chunk = self.backbone(input_cat)
            return self._forward_branch(chunk)
        elif  self.arch ==  'TDNresnet':
            #input_cat = torch.cat(input, dim=1)# B C*T  H W
            chunk = self.backbone(input)
            return self._forward_branch(chunk)
        else:
            chunk = [self.backbone(input[i]) for i in range(self.K)]
            return self._forward_branch(chunk)

