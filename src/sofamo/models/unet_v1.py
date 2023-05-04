# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, use_bn=True, use_act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              n_filters,
                              kernel_size=k_size,
                              padding=padding,
                              stride=stride)
        self.bn = nn.BatchNorm2d(int(n_filters)) if use_bn else nn.Identity()
        self.act = nn.ReLU(inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Sequential(
            ConvBlock(3, 64, 3, 1, 1),
            ConvBlock(64, 64, 3, 1, 1)            
        )
        self.down0 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1) # 256 -> 128

        self.enc_conv1 = nn.Sequential(
            ConvBlock(64, 128, 3, 1, 1),
            ConvBlock(128, 128, 3, 1, 1),       
        )
        self.down1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1) # 128 -> 64

        self.enc_conv2 = nn.Sequential(
            ConvBlock(128, 256, 3, 1, 1),
            ConvBlock(256, 256, 3, 1, 1),
            ConvBlock(256, 256, 3, 1, 1)            
        )
        self.down2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1) # 64 -> 32

        self.enc_conv3 = nn.Sequential(
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 512, 3, 1, 1),
            ConvBlock(512, 512, 3, 1, 1)            
        )
        self.down3 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1) # 32 -> 16

        self.bottle_neck = nn.Sequential(
            ConvBlock(512, 1024, 1, 1, 0),
            ConvBlock(1024, 512, 1, 1, 0)   
        )

        self.up3 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1) # 16 -> 32
        self.dec_conv3 = nn.Sequential(
            ConvBlock(512*2, 256, 3, 1, 1),
            ConvBlock(256, 256, 3, 1, 1),
            ConvBlock(256, 256, 3, 1, 1),
        )

        self.up2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1) # 32 -> 64
        self.dec_conv2 = nn.Sequential(
            ConvBlock(256*2, 128, 3, 1, 1),
            ConvBlock(128, 128, 3, 1, 1),
            ConvBlock(128, 128, 3, 1, 1),
        )

        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1) # 64 -> 128
        self.dec_conv1 = nn.Sequential(
            ConvBlock(128*2, 64, 3, 1, 1),
            ConvBlock(64, 64, 3, 1, 1),
        )

        self.up0 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1) # 128 -> 256
        self.dec_conv0 = nn.Sequential(
            ConvBlock(64*2, 1, 3, 1, 1),
            ConvBlock(1, 1, 3, 1, 1, use_bn=False, use_act=False),
        )

    def forward(self, x):
        # encoder
        pre_e0 = self.enc_conv0(x)
        e0 = self.down0(pre_e0)
        pre_e1 = self.enc_conv1(e0)
        e1 = self.down1(pre_e1)
        pre_e2 = self.enc_conv2(e1)
        e2 = self.down2(pre_e2)
        pre_e3 = self.enc_conv3(e2)
        e3 = self.down3(pre_e3)

        # bottleneck        
        bottle_neck = self.bottle_neck(e3)

        # decoder       
        d3 = self.dec_conv3(torch.cat([self.up3(bottle_neck, output_size=pre_e3.size()), pre_e3], 1))
        d2 = self.dec_conv2(torch.cat([self.up2(d3, output_size=pre_e2.size()), pre_e2], 1))
        d1 = self.dec_conv1(torch.cat([self.up1(d2, output_size=pre_e1.size()), pre_e1], 1))
        d0 = self.dec_conv0(torch.cat([self.up0(d1, output_size=pre_e0.size()), pre_e0], 1))

        return d0