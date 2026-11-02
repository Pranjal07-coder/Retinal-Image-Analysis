"""
Segmentation models for retinal vessel segmentation.
Includes U-Net and Attention U-Net architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block: Conv-BN-ReLU-Conv-BN-ReLU"""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if dropout > 0:
            self.dropout = nn.Dropout2d(dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.conv(x)
        if self.dropout:
            x = self.dropout(x)
        return x


class AttentionBlock(nn.Module):
    """Attention gate for Attention U-Net"""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: Gating signal from decoder (coarser scale)
            x: Skip connection from encoder (finer scale)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNet(nn.Module):
    """
    Standard U-Net architecture for vessel segmentation.

    Args:
        in_channels: Number of input channels (typically 3 for RGB)
        out_channels: Number of output channels (typically 1 for binary segmentation)
        base_filters: Number of filters in first layer (doubled at each downsample)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_filters: int = 64):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = ConvBlock(base_filters * 4, base_filters * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_filters * 8, base_filters * 16)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_filters * 16, base_filters * 8)

        self.upconv3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_filters * 8, base_filters * 4)

        self.upconv2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_filters * 4, base_filters * 2)

        self.upconv1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_filters * 2, base_filters)

        # Final output
        self.out = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.out(dec1))


class AttentionUNet(nn.Module):
    """
    Attention U-Net architecture for vessel segmentation.
    Uses attention gates to focus on relevant features.

    Args:
        in_channels: Number of input channels (typically 3 for RGB)
        out_channels: Number of output channels (typically 1 for binary segmentation)
        base_filters: Number of filters in first layer
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_filters: int = 64):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = ConvBlock(base_filters * 4, base_filters * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock(base_filters * 8, base_filters * 16)

        # Decoder with attention
        self.upconv4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=base_filters * 8, F_l=base_filters * 8, F_int=base_filters * 4)
        self.dec4 = ConvBlock(base_filters * 16, base_filters * 8)

        self.upconv3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=base_filters * 4, F_l=base_filters * 4, F_int=base_filters * 2)
        self.dec3 = ConvBlock(base_filters * 8, base_filters * 4)

        self.upconv2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=base_filters * 2, F_l=base_filters * 2, F_int=base_filters)
        self.dec2 = ConvBlock(base_filters * 4, base_filters * 2)

        self.upconv1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=base_filters, F_l=base_filters, F_int=base_filters // 2)
        self.dec1 = ConvBlock(base_filters * 2, base_filters)

        # Final output
        self.out = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder with attention
        dec4 = self.upconv4(bottleneck)
        enc4_att = self.att4(g=dec4, x=enc4)
        dec4 = torch.cat([dec4, enc4_att], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        enc3_att = self.att3(g=dec3, x=enc3)
        dec3 = torch.cat([dec3, enc3_att], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        enc2_att = self.att2(g=dec2, x=enc2)
        dec2 = torch.cat([dec2, enc2_att], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        enc1_att = self.att1(g=dec1, x=enc1)
        dec1 = torch.cat([dec1, enc1_att], dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.out(dec1))


if __name__ == "__main__":
    # Test models
    print("Testing U-Net...")
    unet = UNet(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 512, 512)
    output = unet(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in unet.parameters()):,}")

    print("\nTesting Attention U-Net...")
    att_unet = AttentionUNet(in_channels=3, out_channels=1)
    output = att_unet(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in att_unet.parameters()):,}")
