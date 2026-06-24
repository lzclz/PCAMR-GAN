import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model1 import common
from utils.tools_3D import extract_image_patches, reduce_sum, same_padding

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, padding=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x0):
        avg_out = torch.mean(x0, dim=1, keepdim=True)
        max_out, _ = torch.max(x0, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x0 * self.sigmoid(x)

# class PyramidAttention(nn.Module):
#     def __init__(self, channel, level=5, res_scale=1, reduction=2, ksize=3, stride=1, softmax_scale=10, average=True,
#                  conv=common.default_conv):
#         super(PyramidAttention, self).__init__()
#         self.channel = channel
#         self.ksize = ksize
#         self.stride = stride
#         self.res_scale = res_scale
#         self.softmax_scale = softmax_scale
#         self.scale = [1 - i / 10 for i in range(level)]
#         self.average = average
#         escape_NaN = torch.FloatTensor([1e-4])
#         self.register_buffer('escape_NaN', escape_NaN)
#
#         self.conv_match_L_base = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
#         self.conv_match = common.BasicBlock(conv, channel, channel // reduction, 1, bn=False, act=nn.PReLU())
#         self.conv_assembly = common.BasicBlock(conv, channel, channel, 1, bn=False, act=nn.PReLU())
#     def forward(self, input):
#         res = input
#         # theta，将相关操作转换为三维版本，用于特征变换
#         match_base = self.conv_match_L_base(input)
#         shape_base = list(res.size())
#         input_groups = torch.split(match_base, 1, dim=0)
#         # patch size for matching，这里ksize在三维下要考虑三个维度的大小
#         # raw_w 用于后续重建，存储不同尺度下的特征信息（三维情况）
#         kernel = self.ksize
#         raw_w = []
#         # w 用于匹配操作，同样存储不同尺度下的特征信息（三维情况）
#         w = []
#         # build feature pyramid，构建特征金字塔，处理三维特征图的不同尺度
#         for i in range(len(self.scale)):
#             ref = input
#             if self.scale[i]!= 1:
#                 # 三维插值，调整特征图的大小（深度、高度、宽度方向），需使用合适的插值模式，这里示例仍用双三次插值（可能需根据实际优化）
#                 ref = F.interpolate(input, scale_factor=self.scale[i], mode='trilinear')
#             # 特征变换函数 f，转换为三维卷积等操作
#             base = self.conv_assembly(ref)
#             shape_input = base.shape
#             # 三维采样操作，提取图像块，需调整参数以适应三维情况
#             raw_w_i = extract_image_patches(base, ksizes=[kernel, kernel, kernel],
#                                                strides=[self.stride, self.stride, self.stride],
#                                                rates=[1, 1, 1],
#                                                padding='same')
#             raw_w_i = raw_w_i.view(shape_input[0], shape_input[1], self.ksize, self.ksize, self.ksize, -1)
#             raw_w_i = raw_w_i.permute(0, 5, 1, 2, 3, 4)  # raw_shape: [N, L, C, k, k, k]，调整维度顺序以方便后续操作
#             raw_w_i_groups = torch.split(raw_w_i, 1, dim=0)
#             raw_w.append(raw_w_i_groups)
#
#             # 特征变换函数 g，同样转换为三维操作
#             ref_i = self.conv_match(ref)
#             shape_ref = ref_i.shape
#             # 三维采样，提取图像块用于匹配
#             w_i = extract_image_patches(ref_i, ksizes=[self.ksize, self.ksize, self.ksize],
#                                                strides=[self.stride, self.stride, self.stride],
#                                                rates=[1, 1, 1],
#                                                padding='same')
#             w_i = w_i.view(shape_ref[0], shape_ref[1], self.ksize, self.ksize, self.ksize, -1)
#             w_i = w_i.permute(0, 5, 1, 2, 3, 4)  # w shape: [N, L, C, k, k, k]
#             w_i_groups = torch.split(w_i, 1, dim=0)
#             w.append(w_i_groups)
#
#         y = []
#         for idx, xi in enumerate(input_groups):
#             # 组内操作，在滤波器内合并不同尺度的特征（三维下的合并）
#             wi = torch.cat([w[i][idx][0] for i in range(len(self.scale))], dim=0)  # [L, C, k, k, k]
#             # 归一化操作，对三维特征进行归一化，计算范数等操作要考虑三维维度
#             max_wi = torch.max(torch.sqrt(reduce_sum(torch.pow(wi, 2),
#                                                      axis=[1, 2, 3, 4],
#                                                      keepdim=True)),
#                                self.escape_NaN)
#             wi_normed = wi / max_wi
#             # 匹配操作，三维卷积进行匹配，调整输入维度等以适配三维卷积
#             xi = same_padding(xi, [self.ksize, self.ksize, self.ksize], [1, 1, 1], [1, 1, 1])  # xi: 1*c*D*H*W
#             yi = F.conv3d(xi, wi_normed, stride=1)  # [1, L, D, H, W] ，L = shape_ref[2]*shape_ref[3]*shape_ref[4]
#             yi = yi.view(1, wi.shape[0], shape_base[2], shape_base[3], shape_base[4])  # (B=1, C=..., D=..., H=..., W=...)
#             # 软最大匹配得分，在指定维度上应用softmax
#             yi = F.softmax(yi * self.softmax_scale, dim=1)
#
#             if self.average == False:
#                 yi = (yi == yi.max(dim=1, keepdim=True)[0]).float()
#
#             # 转置卷积用于特征拼接（三维转置卷积操作）
#             raw_wi = torch.cat([raw_w[i][idx][0] for i in range(len(self.scale))], dim=0)
#             yi = F.conv_transpose3d(yi, raw_wi, stride=self.stride, padding=1) / 8.  # 这里除以8是一种示例调整，可能需根据实际验证优化
#             y.append(yi)
#
#         y = torch.cat(y, dim=0) + res * self.res_scale  # 回到小批量维度，加上残差连接并应用缩放
#         return torch.relu(y)


class PyramidAttention(nn.Module):
    def __init__(self, in_channels, num_scales=3, reduction=16):
        super(PyramidAttention, self).__init__()
        self.num_scales = num_scales
        # 使用卷积来获取不同尺度的信息，每个卷积层的输出通道数保持为 in_channels
        self.conv_scales = nn.ModuleList([
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1)
            for _ in range(num_scales)
        ])
        # 通道注意力
        self.channel_attention = ChannelAttention(in_channels, reduction)
        # 空间注意力
        self.spatial_attention = SpatialAttention()
        # 用于将多尺度拼接后的通道数降到 in_channels
        self.conv_fuse = nn.Conv3d(in_channels * num_scales, in_channels, kernel_size=1)
    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        # 计算不同尺度的特征图
        scale_features = [conv(x) for conv in self.conv_scales]
        # 将多尺度特征图合并
        scale_features = torch.cat(scale_features, dim=1)  # [batch_size, in_channels * num_scales, depth, height, width]
        # 使用 1x1 卷积将通道数降到 in_channels
        scale_features = self.conv_fuse(scale_features)
        # 通道注意力机制
        scale_features = self.channel_attention(scale_features)
        # 空间注意力机制
        scale_features = self.spatial_attention(scale_features)
        return scale_features


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # 通道注意力的结构
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # 计算通道注意力
        avg_out = self.avg_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)
        avg_out = self.sigmoid(avg_out)
        return x * avg_out
class CARAFE(nn.Module):
    def __init__(self, inC, outC, kernel_size=3, up_factor=2):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        # 将二维卷积层改为三维卷积层，用于通道数的降维
        self.down = nn.Conv3d(inC, inC // 4, 1)
        # 调整卷积核大小等参数，构建用于生成卷积核张量的三维卷积层
        self.encoder = nn.Conv3d(inC // 4, self.up_factor ** 3 * self.kernel_size ** 3,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv3d(inC, outC, 1)
        self.prelu = nn.ReLU()
    def forward(self, in_tensor):
        N, C, D, H, W = in_tensor.size()
        kernel_tensor = self.down(in_tensor)  # (N, Cm, D, H, W)，对通道数进行降维
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^3 * Kup^3, D, H, W)，生成用于后续上采样的卷积核张量
        kernel_tensor = pixel_shuffle_3d(kernel_tensor, self.up_factor)  # (N, S^3 * Kup^3, D, H, W)->(N, Kup^3, S*D, S*H, S*W)，三维像素重排
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^3, S*D, S*H, S*W)，在卷积核维度上进行softmax归一化
        # 以下是对卷积核张量进行维度展开和调整操作，使其符合后续计算要求
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)  # (N, Kup^3, D, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)  # (N, Kup^3, D, H, S, S)
        kernel_tensor = kernel_tensor.unfold(4, self.up_factor, step=self.up_factor)  # (N, Kup^3, D, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 3, D, H, W, self.up_factor ** 3)  # (N, Kup^3, D, H, W, S^3)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 4, 1, 5)  # (N, D, H, W, Kup^3, S^3)
        # content-aware reassembly module（内容感知重组装模块）
        # 对输入张量进行填充操作，以适应后续基于卷积核的特征提取
        in_tensor = F.pad(in_tensor, pad=(self.kernel_size // 2, self.kernel_size // 2,
                                          self.kernel_size // 2, self.kernel_size // 2,
                                          self.kernel_size // 2, self.kernel_size // 2),
                          mode='constant', value=0)  # (N, C, D+Kup//2+Kup//2, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        # 对填充后的输入张量进行维度展开操作，以提取局部特征
        in_tensor = in_tensor.unfold(2, self.kernel_size, step=1)  # (N, C, D, W+Kup//2+Kup//2, Kup)
        in_tensor = in_tensor.unfold(3, self.kernel_size, step=1)  # (N, C, D, H, Kup, Kup)
        in_tensor = in_tensor.unfold(4, self.kernel_size, step=1)  # (N, C, D, H, W, Kup, Kup)
        in_tensor = in_tensor.reshape(N, C, D, H, W, -1)  # (N, C, D, H, W, Kup^3)
        in_tensor = in_tensor.permute(0, 2, 3, 4, 1, 5)  # (N, D, H, W, C, Kup^3)
        # 通过矩阵乘法将输入特征与卷积核进行融合，实现内容感知的上采样
        out_tensor = torch.matmul(in_tensor, kernel_tensor)  # (N, D, H, W, C, S^3)
        out_tensor = out_tensor.reshape(N, D, H, W, -1)
        out_tensor = out_tensor.permute(0, 4, 1, 2, 3)
        out_tensor = pixel_shuffle_3d(out_tensor, self.up_factor)
        out_tensor = self.prelu(self.out(out_tensor))
        return out_tensor
def pixel_shuffle_3d(input, upscale_factor):
    batch_size, channels, depth, height, width = input.size()
    # 假设 channels 是 S^3 * K_{up}^3
    channels_per_group = upscale_factor ** 3
    # 计算出每个维度应该如何拆分
    if channels % channels_per_group != 0:
        raise ValueError(f"Channels {channels} must be divisible by {channels_per_group}.")
    # 拆分通道维度，将 S^3 * K_{up}^3 拆分为 S^3 和 K_{up}^3
    input = input.view(batch_size, channels // channels_per_group, channels_per_group,
                        depth, height, width)
    # 使用 pixel shuffle 操作进行空间维度的扩展
    output = input.permute(0, 1, 3, 4, 5, 2).contiguous()
    output = output.view(batch_size, channels // channels_per_group,
                         depth * upscale_factor, height * upscale_factor, width * upscale_factor)
    return output

class Generator(nn.Module):
    def __init__(self, scale_factor):
        # 上采样块数，8倍就有3个（这里简化处理，三维也按类似二维整体缩放比例计算）
        upsample_block_num = int(math.log(scale_factor, 2))
        super(Generator, self).__init__()
        # 连接卷积层和激活函数层
        # 3个通道改为适配三维数据的通道数（假设三维数据的通道数还是3，可根据实际修改），64个卷积核，卷积大小为 (3, 3, 3)，填充合适值来保持尺寸，这里填充 (1, 1, 1)
        self.block10 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.PReLU()
        )
        # 以下各个模块依次修改为三维版本，这里假设对应的自定义模块（ResidualBlock、PyramidAttention、CARAFE）已经有合适的三维实现
        self.block11 = nn.Sequential(ResidualBlock(64), CARAFE(64, 64))  # 2
        self.block2 = nn.Sequential(ResidualBlock(64))  # 4
        # 修改池化层为三维最大池化，核大小和步长调整为三维情况
        self.block3 = nn.Sequential(CARAFE(64, 64), ResidualBlock(64),
                                    nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))  # 2
        self.block4 = nn.Sequential(ResidualBlock(128),
                                    nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.block5 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64)
        )
        # 上采样层
        block8 = [nn.Sequential(ResidualBlock(64), CARAFE(64, 64)) for _ in range(upsample_block_num)]
        block8.append(nn.Sequential(ResidualBlock(64)))
        block8.append(nn.Sequential(nn.Conv3d(64, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1))))
        self.block8 = nn.Sequential(*block8)
    def forward(self, x):
        block10 = self.block10(x)  # 1
        block11 = self.block11(block10)  # 2
        block20 = self.block2(block11)  # 2
        block30 = self.block3(block20)  # 4
        block31 = torch.cat((block30, block20), 1)
        block4 = self.block4(block31)  # 2
        block5 = self.block5(block4)  # 1
        block8 = self.block8((block10 + block5))
        return (torch.tanh(block8) + 1) / 2


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # 将二维批量归一化层nn.BatchNorm2d修改为三维批量归一化层nn.BatchNorm3d，用于对三维特征图进行归一化操作
        self.bn1 = nn.BatchNorm3d(channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(channels)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv3d(channels, channels, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(channels)
        self.prelu3 = nn.PReLU()
    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu1(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.prelu2(residual)
        residual = self.conv3(residual)
        residual = self.bn3(residual)
        residual = self.prelu3(residual)
        # 将输入特征图与残差部分相加，并通过ReLU激活函数（这里可根据实际情况选择激活函数，示例中沿用原代码的思路使用ReLU）
        return torch.relu(x + residual)


# 这里假设ResidualBlock1和SpatialAttention模块已经有对应的三维实现或者可以按照类似思路修改为三维版本，以下先给出类定义框架，具体模块内部实现需根据实际情况调整
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.block1 = nn.Sequential(
            # 3个通道，8个卷积核，卷积大小在三维下变为 (3, 3, 3)，填充方式相应调整，使用谱归一化
            nn.utils.spectral_norm(nn.Conv3d(3, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1))),
            nn.ReLU()
        )
        self.block2 = ResidualBlock1(8, 16)  # 假设ResidualBlock1已修改为三维版本能处理合适输入输出通道数
        self.block3 = ResidualBlock1(16, 32)
        self.block41 = ResidualBlock1(32, 64)
        self.block42 = ResidualBlock1(64, 128)
        self.block43 = ResidualBlock1(248, 256)
        self.pam = SpatialAttention()  # 假设SpatialAttention已修改为三维版本
        self.block5 = nn.AdaptiveAvgPool3d(1)  # 改为三维平均池化，输出尺寸为1的三维张量
        self.block6 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv3d(256, 512, kernel_size=1)),
            nn.ReLU(),
            (nn.Conv3d(512, 1, kernel_size=1))
        )
    def forward(self, x):
        batch_size = x.size(0)
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block41 = self.block41(block3)
        block42 = self.block42(block41)
        # 在通道维度上拼接，注意维度顺序需符合三维张量要求
        block4 = torch.cat((block1, block2, block3, block41, block42), 1)
        block43 = self.block43(block4)
        block51 = self.pam(block43)
        block52 = self.block5(block51)
        block6 = self.block6(block52)
        return torch.sigmoid(block6.view(batch_size, -1))

class ResidualBlock1(nn.Module):
    def __init__(self, channels,outchannels):
        super(ResidualBlock1, self).__init__()
        self.conv2 = nn.utils.spectral_norm(nn.Conv3d(channels, outchannels, kernel_size=3, padding=1))
        self.conv3 = nn.utils.spectral_norm(nn.Conv3d(outchannels, outchannels, kernel_size=3, padding=1))
        self.prelu2 =nn.LeakyReLU(0.2)
        self.prelu3 =nn.ReLU()
    def forward(self, x):
        residual = self.conv2(x)
        residual = self.prelu2(residual)
        residual = self.conv3(residual)
        residual = self.prelu3(residual)
        return residual
