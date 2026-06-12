import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



# -----Shared building block-----


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)



# -----Segmentation: ResU-Net (resnet34 / resnet50 backbone)------


class ResUNet(nn.Module):
    def __init__(self, n_classes: int = 1, backbone: str = "resnet34", pretrained: bool = True):
        super().__init__()
        if pretrained:
            weights = getattr(models, backbone.replace("resnet", "ResNet") + "_Weights").DEFAULT
        else:
            weights = None
        resnet = getattr(models, backbone)(weights=weights)

        self.input_layer = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4

        exp = getattr(self.layer1[0], "expansion", 1)
        c0, c2, c3, c4, c5 = 64, 64*exp, 128*exp, 256*exp, 512*exp
        d4, d3, d2, d1 = max(c4//2, 64), max(c3//2, 64), max(c2//2, 32), 64

        self.up4  = nn.ConvTranspose2d(c5, d4, 2, 2);  self.dec4 = DoubleConv(d4 + c4, d4)
        self.up3  = nn.ConvTranspose2d(d4, d3, 2, 2);  self.dec3 = DoubleConv(d3 + c3, d3)
        self.up2  = nn.ConvTranspose2d(d3, d2, 2, 2);  self.dec2 = DoubleConv(d2 + c2, d2)
        self.up1  = nn.ConvTranspose2d(d2, d1, 2, 2);  self.dec1 = DoubleConv(d1 + c0, 64)
        self.out_conv = nn.Conv2d(64, n_classes, 1)

    def _match(self, a, b):
        if a.shape[-2:] == b.shape[-2:]:
            return a
        return F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        x0 = self.input_layer(x)
        x1 = self.maxpool(x0)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        d4 = self.dec4(torch.cat([self._match(self.up4(x5), x4), x4], 1))
        d3 = self.dec3(torch.cat([self._match(self.up3(d4), x3), x3], 1))
        d2 = self.dec2(torch.cat([self._match(self.up2(d3), x2), x2], 1))
        d1 = self.dec1(torch.cat([self._match(self.up1(d2), x0), x0], 1))

        out = self.out_conv(d1)
        return F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)



# -----Classification: Siamese ResNet (late fusion)-----


class SiameseResNet(nn.Module):
    """Late-fusion Siamese network: encodes pre/post independently, fuses via [f0, f1, |f1-f0|, f0*f1]."""

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True,
                 num_classes: int = 4, dropout: float = 0.4):
        super().__init__()
        if backbone == "resnet50":
            w = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=w)
        elif backbone == "resnet34":
            w = models.ResNet34_Weights.DEFAULT if pretrained else None
            base = models.resnet34(weights=w)
        else:
            raise ValueError("backbone must be 'resnet50' or 'resnet34'")
        feat_dim = base.fc.in_features
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(4 * feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def encode(self, x):
        return self.encoder(x).flatten(1)

    def forward(self, pre, post):
        f0, f1 = self.encode(pre), self.encode(post)
        z = torch.cat([f0, f1, torch.abs(f1 - f0), f0 * f1], dim=1)
        return self.head(z)



# -----Classification: Early-Fusion ResNet (6-channel input)-----


class EarlyFusionResNet(nn.Module):
    """Early-fusion: concatenates pre+post (and optionally |post-pre|) into a 6- or 9-channel input."""

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True,
                 num_classes: int = 4, dropout: float = 0.4, add_diff: bool = False):
        super().__init__()
        if backbone == "resnet50":
            w = models.ResNet50_Weights.DEFAULT if pretrained else None
            base = models.resnet50(weights=w)
        elif backbone == "resnet34":
            w = models.ResNet34_Weights.DEFAULT if pretrained else None
            base = models.resnet34(weights=w)
        else:
            raise ValueError("backbone must be 'resnet50' or 'resnet34'")

        in_ch = 6 + (3 if add_diff else 0)
        old = base.conv1
        new = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=(old.bias is not None))
        with torch.no_grad():
            chunks = in_ch // 3
            for k in range(chunks):
                new.weight[:, 3*k:3*(k+1), :, :] = old.weight / chunks
            if old.bias is not None and new.bias is not None:
                new.bias.copy_(old.bias)
        base.conv1 = new

        feat_dim = base.fc.in_features
        self.encoder = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self.add_diff = add_diff

    def forward(self, pre, post):
        x = torch.cat([pre, post, torch.abs(post - pre)], dim=1) if self.add_diff else torch.cat([pre, post], dim=1)
        return self.head(self.encoder(x).flatten(1))



# -----Losses-----


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class alpha weighting."""

    def __init__(self, alpha=None, gamma: float = 1.5, reduction: str = "mean"):
        super().__init__()
        if alpha is None:
            self.alpha = None
        else:
            a = torch.tensor(alpha, dtype=torch.float32)
            self.alpha = a / (a.sum() / len(a))
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        ce   = F.nll_loss(logp, targets, reduction="none")
        pt   = torch.exp(-ce).clamp_min(1e-9)
        fl   = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            fl = self.alpha.to(logits.device)[targets] * fl
        return fl.mean() if self.reduction == "mean" else fl.sum()


class _DummyScaler:
    """No-op GradScaler for CPU / non-AMP runs."""
    def scale(self, loss):      return loss
    def step(self, opt):        opt.step()
    def update(self):           pass
    def unscale_(self, opt):    pass
