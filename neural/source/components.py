"""Untrained CPU-verifiable design sketch. No denoising output or live integration.

NAF-inspired local blocks: SimpleGate, channel LayerNorm, zero-scaled residuals.
Deliberately omit the original global SCA inside each pixel block; global
interaction happens once on regional tokens. This is not the original NAFNet.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ChannelNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        var = (x - mean).square().mean(1, keepdim=True)
        return (x - mean) * torch.rsqrt(var + 1e-6) * self.weight + self.bias


class Gate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, 1)
        return a * b


class LocalNAFBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm1, self.norm2 = ChannelNorm(c), ChannelNorm(c)
        self.branch1 = nn.Sequential(
            nn.Conv2d(c, 2*c, 1), nn.Conv2d(2*c, 2*c, 3, padding=1, groups=2*c),
            Gate(), nn.Conv2d(c, c, 1))
        self.branch2 = nn.Sequential(nn.Conv2d(c, 2*c, 1), Gate(), nn.Conv2d(c, c, 1))
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        y = x + self.beta * self.branch1(self.norm1(x))
        return y + self.gamma * self.branch2(self.norm2(y))


class SpatialEncoder(nn.Module):
    """Inputs: 12 fixed-scale signed feature planes. Downsample factor 8.

    PixelUnshuffle is lossless; its subsequent channel projection is not.
    Widths 24/48/96/128, blocks 1/1/2/2; receptive field 64 packed samples.
    Test tiling on already padded whole inputs with origins aligned to 8.
    """
    def __init__(self):
        super().__init__()
        layers = [nn.Conv2d(12, 24, 3, padding=1), LocalNAFBlock(24)]
        for a, b, n in ((24, 48, 1), (48, 96, 2), (96, 128, 2)):
            layers += [nn.PixelUnshuffle(2), nn.Conv2d(4*a, b, 1)]
            layers += [LocalNAFBlock(b) for _ in range(n)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def region_statistics(raw, rn, saturation, region=64):
    """58 descriptors per complete region, for test-sized divisible inputs.

    Production needs masked edge regions: padding may not count as observations.
    Soft CDF widths/thresholds are fixed by RN; no per-image centering.
    """
    assert raw.shape[-2] % region == 0 and raw.shape[-1] % region == 0
    pool = lambda z: F.avg_pool2d(z, region, region)
    mean = pool(raw)
    var = (pool(raw.square()) - mean.square()).clamp_min(0)
    negative = pool((raw < 0).to(raw.dtype))
    saturated = pool((raw >= saturation).to(raw.dtype))
    gx = F.pad((raw[..., 1:] - raw[..., :-1]).abs(), (0, 1))
    gy = F.pad((raw[..., 1:, :] - raw[..., :-1, :]).abs(), (0, 0, 0, 1))
    gradient = pool((gx + gy)/2)
    z = raw / rn
    cdf = [pool(torch.sigmoid((t-z)/0.25))
           for t in (-4., -2., -1., -.5, 0., .5, 1., 2., 4.)]
    count = torch.full_like(mean[:, :1], float(region*region)).log()
    valid = torch.ones_like(count)
    return torch.cat([mean/64, torch.log1p(var), negative, saturated,
                      torch.log1p(gradient), *cdf, count, valid], 1)


class FrameAggregator(nn.Module):
    """128 learned feature mean+std, 58 stats, 17 metadata, 2 position =333."""
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(333, 192)
        self.queries = nn.Parameter(torch.randn(1, 8, 192)*.02)
        self.attention = nn.MultiheadAttention(192, 4, batch_first=True)
        self.out = nn.Sequential(nn.Linear(8*192+2*192, 256), nn.SiLU())
        from bounded_head import BoundedDNHead
        self.point = BoundedDNHead(256, 4)

    def forward(self, descriptors, valid=None):
        t = self.project(descriptors)
        q = self.queries.expand(t.shape[0], -1, -1)
        a, _ = self.attention(q, t, t, key_padding_mask=None if valid is None else ~valid,
                              need_weights=False)
        w = torch.ones_like(t[..., :1]) if valid is None else valid[..., None].to(t.dtype)
        mean = (t*w).sum(1) / w.sum(1).clamp_min(1)
        var = ((t-mean[:, None]).square()*w).sum(1) / w.sum(1).clamp_min(1)
        f = self.out(torch.cat([a.flatten(1), mean, (var+1e-6).sqrt()], -1))
        return f, self.point(f)


class TemporalHead(nn.Module):
    """Shared two-layer GRU for forward/reverse scans; direction is explicit.

    Offline fusion learns correlations rather than multiplying independent
    frame likelihoods. For a one-frame sequence return the spatial prediction.
    """
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(258, 192, 2, batch_first=True)
        self.causal = nn.Sequential(nn.Linear(256+192, 128), nn.SiLU(), nn.Linear(128, 4))
        self.offline = nn.Sequential(nn.Linear(256+384, 128), nn.SiLU(), nn.Linear(128, 4))
        for head in (self.causal, self.offline):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def scan(self, f, dt, direction=1., state=None):
        d = torch.full_like(dt, direction)
        return self.rnn(torch.cat([f, torch.log(dt.clamp_min(1e-6)), d], -1), state)

    def stream(self, f, point, dt, state=None):
        cold_start = state is None
        h, state = self.scan(f, dt, state=state)
        out = point + self.causal(torch.cat([f, h], -1))
        if cold_start:
            out = torch.cat([point[:, :1], out[:, 1:]], 1)
        return out, state

    def sequence(self, f, point, dt, offline=False):
        if f.shape[1] == 1:
            return point
        if not offline:
            return self.stream(f, point, dt)[0]
        hf, _ = self.scan(f, dt)
        # Reversal must use the interval to the NEXT frame, not preceding one.
        dt_next = torch.cat([dt[:, 1:], dt[:, -1:]], 1)
        hb, _ = self.scan(f.flip(1), dt_next.flip(1), -1.)
        return point+self.offline(torch.cat([f, hf, hb.flip(1)], -1))


class DesignPrototype(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = SpatialEncoder()
        self.frame = FrameAggregator()
        self.temporal = TemporalHead()


def estimation_loss(frame, causal, offline, truth):
    """Initial frozen design coefficients, not empirically selected hyperparams.
    Average over valid entries; caller must mask cuts/missing labels explicitly.
    beta=.05 DN. Drift supervision matches true increments, not zero increments.
    """
    huber = lambda a, b: F.smooth_l1_loss(a, b, beta=.05)
    point = .25*huber(frame, truth)+.375*huber(causal, truth)+.375*huber(offline, truth)
    delta = lambda x: x[:, 1:]-x[:, :-1]
    if truth.shape[1] < 2:
        return huber(frame, truth)
    drift = .5*huber(delta(causal), delta(truth))+.5*huber(delta(offline), delta(truth))
    return point + .1*drift
