# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.utils.checkpoint import checkpoint

if __name__ == "__main__":
    from .blocks.ASSP import ASSP
else:
    from sofamo.models.blocks.ASSP import ASSP


# %%
def initialize_weights(modules):
    for m in modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.fill_(1e-4)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0.0, 0.0001)
            m.bias.data.zero_()


activations = nn.ModuleDict(
    {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.05, inplace=True),
        "prelu": nn.PReLU(),
        "gelu": nn.GELU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
    }
)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation="relu",
        kernel_size=3,
        padding=1,
        dilation=1,
        use_bn=False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.conv0 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn0 = nn.BatchNorm2d(out_channels) if use_bn else None
        self.act0 = activations[str(activation)] if activation else None
        self.conv1 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else None
        self.act1 = activations[str(activation)] if activation else None

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x) if self.bn0 is not None else x
        x = self.act0(x) if self.act0 is not None else x
        x = self.conv1(x)
        x = self.bn1(x) if self.bn1 is not None else x
        x = self.act1(x) if self.act1 is not None else x
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, activation, use_bn):
        super().__init__()
        self.double_conv = ConvBlock(
            in_channels,
            out_channels,
            activation=activation,
            use_bn=use_bn,
        )
        self.enc_down = nn.MaxPool2d(kernel_size=2, ceil_mode=True)

    def forward(self, x):
        x = self.double_conv(x)
        x_downed = self.enc_down(x)
        return x, x_downed


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, activation, use_bn):
        super().__init__()
        self.dec_up = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=2,
            stride=2,
        )
        self.double_conv = ConvBlock(
            in_channels + out_channels,
            out_channels,
            activation=activation,
            use_bn=use_bn,
        )

    def forward(self, x_transfered, x, interpolate=True):
        x_upped = self.dec_up(x)

        if interpolate:
            # Iterpolating instead of padding gives better results
            size2 = x_transfered.size()[2]
            size3 = x_transfered.size()[3]
            x_upped = F.interpolate(
                x_upped,
                size=(size2, size3),
                mode="bilinear",
                align_corners=True,
            )
        else:
            # Padding in case the incomping volumes are of different sizes
            diff_y = x_transfered.size()[2] - x_upped.size()[2]
            diff_x = x_transfered.size()[3] - x_upped.size()[3]
            pad_x = diff_x // 2, diff_x - diff_x // 2
            pad_y = diff_y // 2, diff_y - diff_y // 2
            x_upped = F.pad(x_upped, (pad_x, pad_y))

        # Concatenate
        # print(f"In decoder. x.transfered.shape: {x_transfered.shape}, x_upped.shape: {x_upped.shape}")
        x = torch.cat([x_upped, x_transfered], dim=1)
        x = self.double_conv(x)
        return x


# %%
class UNet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        deep_channels=[32, 64, 128, 256],
        activation="relu",
        use_bn=False,
        use_assp=False,
        use_dropout=False,
        is_features_return=False,
        **_,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.is_features_return = is_features_return
        self.use_assp = use_assp

        middle_channels = deep_channels[-1]
        deep_channels = deep_channels[:-1]
        enc_channels = [in_channels, *deep_channels]
        dec_channels = [middle_channels, *deep_channels[::-1]]

        self.down = nn.ModuleList()
        self.middle = nn.ModuleList()
        self.up = nn.ModuleList()

        for n_in, n_out in zip(enc_channels[:-1], enc_channels[1:]):
            # print(f"Encoder. n_in: {n_in}, n_out: {n_out}")
            self.down.append(Encoder(n_in, n_out, activation, use_bn))

        if use_assp:
            self.middle.append(
                ASSP(
                    in_channels=n_out,
                    out_channels=middle_channels,
                    kernel_sizes=[1, 3, 3, 3],
                    dilations=[1, 6, 12, 24],
                    use_bn=use_bn,
                )
            )
        else:
            self.middle.append(
                ConvBlock(
                    n_out,
                    middle_channels,
                    activation=activation,
                    use_bn=use_bn,
                )
            )

        for n_in, n_out in zip(dec_channels[:-1], dec_channels[1:]):
            # print(f"Decoder. n_in: {n_in}, n_out: {n_out}")
            self.up.append(Decoder(n_in, n_out, activation, use_bn))

        if self.out_channels is not None:
            self.final_conv = nn.Conv2d(n_out, out_channels, kernel_size=1)

    def forward(self, x):
        transfer = []
        for i, down_layer in enumerate(self.down):
            x_trasfered, x = down_layer(x)
            # print(f"x_trasfered.shape: {x_trasfered.shape}, x_downed.shape : {x.shape}")
            transfer.append(x_trasfered)
        transfer.reverse()
        for i, middle_layer in enumerate(self.middle):
            x = middle_layer(x)
            # print(f"x.middle.shape: {x.shape}")
        for i, up_layer in enumerate(self.up):
            # print(f"x.transfered.shape: {transfer[i].shape}, x.shape: {x.shape}")
            x = up_layer(transfer[i], x)
            # print(f"x.shape: {x.shape}")

        if self.is_features_return:
            x_features = x
        else:
            x_features = None

        if self.out_channels is not None:
            x_final = self.final_conv(x)
        else:
            x_final = None

        if self.is_features_return:
            return x_final, x_features
        else:
            return x_final



# %%
if __name__ == "__main__":

    model = UNet(
        in_channels=3,
        out_channels=7,
        deep_channels=[128, ] * 4,
        activation="relu",
        use_bn=False,
        use_assp=True,
        is_features_return=True,
    )

    x_dumb = torch.rand([1, 3, 512, 512])
    print(x_dumb.shape)
    y_pred, features = model(x_dumb)
    print(y_pred.shape)
    print(features.shape)

    # def count_parameters(model, is_printing=True):
    #     """
    #     Calculation of train parameters
    #     return (all_params, trainable_params)
    #     """
    #     all_params = sum(p.numel() for p in model.parameters())
    #     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #     if is_printing:
    #         string = f"{trainable_params/1e+6:.2f}M from {all_params/1e+6}M are requires gradients"
    #         print(string)
    #     else:
    #         return all_params, trainable_params

    # count_parameters(model, True)

    import torchsummary

    # # x0_dumb = torch.rand([1, 3, 512, 512])
    # # x1_dumb = torch.rand([1, 3, 512, 512])
    torchsummary.summary(model, (3, 512, 512))
