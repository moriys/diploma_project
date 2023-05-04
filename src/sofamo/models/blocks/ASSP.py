# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
class AsspDilationBranch(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dilation,
                 use_bn=True):
        super().__init__()
        padding = 0 if kernel_size == 1 else dilation
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) if self.bn is not None else x
        x = self.relu(x)
        return x


# %%
class AsspAvgPoolingBranch(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bn=False):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_size = x.size()  # x.shape
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.bn(x) if self.bn is not None else x
        x = self.relu(x)
        x = F.interpolate(
            x,
            size=(x_size[2], x_size[3]),
            mode='bilinear',
            align_corners=True,
        )
        return x


# %%
class ASSP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=[1, 3, 3, 3],
                 #  dilations=[1, 12, 24, 36],
                 dilations=[1, 6, 12, 18],
                 use_bn=False,
                 use_dropout=False):
        super().__init__()

        self.use_bn = use_bn
        self.use_dropout = use_dropout

        self.assp_branches = nn.ModuleList()
        for i in range(len(dilations)):
            assp_branch = AsspDilationBranch(
                in_channels,
                out_channels,
                kernel_sizes[i],
                dilations[i]
            )
            self.assp_branches.append(assp_branch)
        assp_branch = AsspAvgPoolingBranch(
            in_channels,
            out_channels,
            use_bn=use_bn
        )
        self.assp_branches.append(assp_branch)

        depths = len(dilations)
        self.conv1 = nn.Conv2d(
            (depths+1)*out_channels,
            out_channels,
            kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else None
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5) if use_dropout else None
        # initialize_weights(self)

    def forward(self, x):
        X = []
        for i, assp_branch in enumerate(self.assp_branches):
            # print(f"assp_branch(x).shape: {assp_branch(x).shape}")
            X.append(assp_branch(x))
        x = self.conv1(torch.cat(X, dim=1))
        x = self.bn1(x) if self.bn1 else x
        x = self.relu(x)
        x = self.dropout(x) if self.dropout else x
        return x


if __name__ == "__main__":
    batch_channels = 1
    in_channels = 300
    out_channels = 300

    assp = AsspDilationBranch(in_channels, out_channels, 3, 6, False)
    x = torch.rand([batch_channels, in_channels, 1, 1])
    y = assp(x)
    print(x.shape)
    print(y.shape)

    # assp = AsspAvgPoolingBranch(in_channels, out_channels, True)
    # x = torch.rand([batch_channels, in_channels, 40, 40])
    # y = assp(x)
    # print(x.shape)
    # print(y.shape)

    # assp = ASSP(in_channels=in_channels,
    #             out_channels=out_channels,
    #             kernel_sizes=[1, 3, 3, 3],
    #             dilations=[1, 6, 12, 24])
    # x = torch.rand([batch_channels, in_channels, 64, 64])
    # y = assp(x)
    # print(x.shape)
    # print(y.shape)
