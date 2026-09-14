from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch import nn
from .branch import IOD_Branch
from .CPCA import (
    ResidualCPCA,
    VisibilityAdaptiveResidualCPCA,
    OracleResidualCPCA)
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
                 use_cpca=False, cpca_reduction=16,
                 cpca_kernel_sizes=(7, 11, 21), cpca_residual_scale=0.1,
                 use_rgam=False, rgam_groups=4, rgam_reduction_c=16,
                 rgam_reduction_s=4, rgam_spatial_size=(18, 18),
                 rgam_residual_scale=0.1, cpca_gate_hidden=32,
                 use_cpca_oracle_gate=False):
        super(STA_Framework, self).__init__()
        self.K = K
        backbone_kwargs = {}
        if arch == 'TEAresnet':
            backbone_kwargs = dict(
                use_rgam=use_rgam, rgam_groups=rgam_groups,
                rgam_reduction_c=rgam_reduction_c,
                rgam_reduction_s=rgam_reduction_s,
                rgam_spatial_size=rgam_spatial_size,
                rgam_residual_scale=rgam_residual_scale)
        self.backbone = backbone[arch](num_layers, K, **backbone_kwargs)
        self.arch = arch
        self.use_cpca = use_cpca
        self.use_cpca_gate = bool(use_cpca and use_cpca_gate)
        self.use_cpca_oracle_gate = bool(
            use_cpca and use_cpca_oracle_gate)
        if self.use_cpca_gate and self.use_cpca_oracle_gate:
            raise ValueError(
                'use_cpca_gate and use_cpca_oracle_gate are mutually exclusive')
        if self.use_cpca:
            cpca_args = dict(
                channels=self.backbone.output_channel,
                reduction=cpca_reduction,
                kernel_sizes=cpca_kernel_sizes,
                residual_scale=cpca_residual_scale)
            if self.use_cpca_oracle_gate:
                self.cpca = OracleResidualCPCA(
                    channels=self.backbone.output_channel,
                    reduction=cpca_reduction,
                    kernel_sizes=cpca_kernel_sizes,
                    alpha=0.1)
            elif self.use_cpca_gate:
                self.cpca = VisibilityAdaptiveResidualCPCA(
                    gate_hidden=cpca_gate_hidden, **cpca_args)
            else:
                self.cpca = ResidualCPCA(**cpca_args)
        self.branch = IOD_Branch(self.backbone.output_channel, arch, head_conv, branch_info, K)
        self.R2D  = nn.Sequential(
            nn.Conv2d(K * self.backbone.output_channel, self.backbone.output_channel,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True))

    def _refine_chunk(self, chunk, oracle_gate=None):
        if not self.use_cpca:
            return chunk, None
        if self.use_cpca_oracle_gate:
            return self.cpca(chunk, oracle_gate, return_gate=True)
        if self.use_cpca_gate:
            # The adaptive wrapper observes all K frames before applying the
            # same CPCA parameters to each frame.
            return self.cpca(chunk, return_gate=True)
        # A single CPCA instance is reused for every frame, so all K frames
        # share the same attention parameters.
        return [self.cpca(feature) for feature in chunk], None

    def _attach_cpca_monitor(self, output, gate):
        if gate is None:
            return output

        batch_size = gate.size(0)
        # These tensors travel with the normal model output, so DataParallel
        # gathers monitoring values from every GPU correctly.
        output['cpca_gate'] = gate.detach().reshape(batch_size)
        scale = (self.cpca.oracle_alpha
                 if self.use_cpca_oracle_gate
                 else self.cpca.residual_scale)
        output['cpca_residual_scale'] = (
            scale.detach().reshape(1).expand(batch_size))
        return output

    def forward(self, input, oracle_gate=None):
        if self.arch == 'I3Dresnet' or self.arch ==  'S3Dresnet' or self.arch =='TAMresnet' or self.arch ==  'MSresnet' or \
                self.arch == 'TEAresnet' or self.arch == 'TINresnet' or self.arch ==  'TSMresnet':
            inputlist = [input[i].unsqueeze(2) for i in range(self.K)]
            input_cat = torch.cat(inputlist, dim=2)# B C T H W
            chunk = self.backbone(input_cat)
            chunk, gate = self._refine_chunk(chunk, oracle_gate)
            output1 = self.branch(chunk)
            output1 = self._attach_cpca_monitor(output1, gate)
            return [output1]
        elif  self.arch ==  'TDNresnet':
            #input_cat = torch.cat(input, dim=1)# B C*T  H W
            chunk = self.backbone(input)
            chunk, gate = self._refine_chunk(chunk, oracle_gate)
            output1 = self.branch(chunk)
            output1 = self._attach_cpca_monitor(output1, gate)
            return [output1]
        else:
            chunk = [self.backbone(input[i]) for i in range(self.K)]
            chunk, gate = self._refine_chunk(chunk, oracle_gate)
            output1 = self.branch(chunk)
            output1 = self._attach_cpca_monitor(output1, gate)
            return [output1]

