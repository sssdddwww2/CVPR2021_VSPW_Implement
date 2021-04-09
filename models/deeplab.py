import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm import SynchronizedBatchNorm2d



class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, fc_dim, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        inplanes = fc_dim
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet':
            low_level_inplanes = 256
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU()
                                       )
        self.lastlast_conv = nn.Sequential(
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
                                              )

        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)
        x = self.lastlast_conv(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_class=21,fc_dim=2048,
                 sync_bn=True, use_softmax=False):
        super(DeepLab, self).__init__()
        if backbone == 'resnet':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.aspp = ASPP(fc_dim, output_stride, BatchNorm)
        self.decoder = Decoder(num_class, backbone, BatchNorm)


    def forward(self, conv_out, segSize=None):
        x = conv_out[-1]
        low_level_feat = conv_out[-4]
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        if segSize is not None: # is True during inference
            x = nn.functional.interpolate(
                x, size=segSize, mode='bilinear', align_corners=False)
            x = nn.functional.softmax(x, dim=1)
        else:
            x = nn.functional.log_softmax(x, dim=1)

        return x

#    def get_1x_lr_params(self):
#        modules = [self.backbone]
#        for i in range(len(modules)):
#            for m in modules[i].named_modules():
#                for key, p in m[1].named_parameters():
#                    if p.requires_grad and (not ('bias' in key)):
#                        yield p
#
#    def get_10x_lr_params(self):
#        modules = [self.aspp, self.decoder]
#        for i in range(len(modules)):
#            for m in modules[i].named_modules():
#                for key, p in m[1].named_parameters():
#                    if p.requires_grad and (not ('bias' in key)):
#                        yield p
#
#    def get_1x_lr_params_bias(self):
#        modules = [self.backbone]
#        for i in range(len(modules)):
#            for m in modules[i].named_modules():
#                for key, p in m[1].named_parameters():
#                    if p.requires_grad and 'bias' in key:
#                        yield p
#
#    def get_10x_lr_params_bias(self):
#        modules = [self.aspp, self.decoder]
#        for i in range(len(modules)):
#            for m in modules[i].named_modules():
#                for key, p in m[1].named_parameters():
#                    if p.requires_grad and 'bias' in key:
#                        yield p








#    def get_1x_lr_params(self):
#        modules = [self.backbone]
#        for i in range(len(modules)):
#            for m in modules[i].named_modules():
#                if self.freeze_bn:
#                    if isinstance(m[1], nn.Conv2d):
#                        for p in m[1].parameters():
#                            if p.requires_grad:
#                                yield p
#                else:
#                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
#                            or isinstance(m[1], nn.BatchNorm2d):
#                        for p in m[1].parameters():
#                            if p.requires_grad:
#                                yield p
#
#    def get_10x_lr_params(self):
#        modules = [self.aspp, self.decoder]
#        for i in range(len(modules)):
#            for m in modules[i].named_modules():
#                if self.freeze_bn:
#                    if isinstance(m[1], nn.Conv2d):
#                        for p in m[1].parameters():
#                            if p.requires_grad:
#                                yield p
#                else:
#                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
#                            or isinstance(m[1], nn.BatchNorm2d):
#                        for p in m[1].parameters():
#                            if p.requires_grad:
#                                yield p
#
if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    print(model)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


