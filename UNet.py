# https://github.com/openai/guided-diffusion/tree/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924
import math
from abc import abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# ============================================================
# Group-equivariant convolutions (basic C4 / D4)
#
# We implement a small, dependency-free building block that can be
# dropped in place of nn.Conv2d for 3x3 kernels.
#
# - For C4 (rotations by 0/90/180/270) and D4 (C4 + flips), a 3x3 kernel
#   is rotation/flip-invariant iff all corners share a parameter, all
#   edges share a parameter, and the center is its own parameter:
#
#       [a b a]
#       [b c b]
#       [a b a]
#
# This ensures: conv(gx) == g conv(x) for g in {rot90, flip+rot90}, as
# long as all other spatial ops are equivariant.
#
# NOTE: AttentionBlock is NOT rotation-equivariant.
# For group-equiv runs, disable attention by default (handled below).
# ============================================================

class GEConv2d3x3(nn.Module):
    """Group-equivariant 3x3 conv via weight tying (C4/D4)."""

    def __init__(self, in_channels, out_channels, stride=1, padding=1, bias=True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.padding = int(padding)

        # Parameters per (out,in): p[...,0]=corners, p[...,1]=edges, p[...,2]=center
        self.p = nn.Parameter(torch.empty(out_channels, in_channels, 3))
        nn.init.kaiming_normal_(self.p, a=0.2)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

    def _build_weight(self):
        a = self.p[..., 0]
        b = self.p[..., 1]
        c = self.p[..., 2]

        w = torch.zeros(
            (self.out_channels, self.in_channels, 3, 3),
            device=self.p.device,
            dtype=self.p.dtype,
        )

        # corners
        w[:, :, 0, 0] = a
        w[:, :, 0, 2] = a
        w[:, :, 2, 0] = a
        w[:, :, 2, 2] = a

        # edges
        w[:, :, 0, 1] = b
        w[:, :, 1, 0] = b
        w[:, :, 1, 2] = b
        w[:, :, 2, 1] = b

        # center
        w[:, :, 1, 1] = c
        return w

    def forward(self, x):
        w = self._build_weight()
        return F.conv2d(x, w, bias=self.bias, stride=self.stride, padding=self.padding)


def make_conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, group="none"):
    """Factory for Conv2d that optionally enforces C4/D4 equivariance.

    Only swaps **3x3** kernels. 1x1 kernels are already invariant to rotations/flips.
    """
    group = (group or "none").upper()
    if group in ("C4", "D4") and int(kernel_size) == 3:
        return GEConv2d3x3(in_channels, out_channels, stride=stride, padding=padding, bias=bias)
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)


class TimestepBlock(nn.Module):
    """Any module where forward() takes timestep embeddings as a second argument."""
    @abstractmethod
    def forward(self, x, emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """Sequential module that passes timestep embeddings to children that support it."""

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class PositionalEmbedding(nn.Module):
    """Computes positional embedding of the timestep"""

    def __init__(self, dim, scale=1):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(x * self.scale, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample(nn.Module):
    def __init__(self, in_channels, use_conv, out_channels=None, group="none"):
        super().__init__()
        self.channels = in_channels
        out_channels = out_channels or in_channels
        if use_conv:
            self.downsample = make_conv2d(in_channels, out_channels, 3, stride=2, padding=1, group=group)
        else:
            assert in_channels == out_channels
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, time_embed=None):
        assert x.shape[1] == self.channels
        return self.downsample(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, use_conv, out_channels=None, group="none"):
        super().__init__()
        self.channels = in_channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = make_conv2d(in_channels, out_channels, 3, padding=1, group=group)

    def forward(self, x, time_embed=None):
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class AttentionBlock(nn.Module):
    """(Not rotation-equivariant.)"""

    def __init__(self, in_channels, n_heads=1, n_head_channels=-1):
        super().__init__()
        self.in_channels = in_channels
        self.norm = GroupNorm32(32, self.in_channels)
        if n_head_channels == -1:
            self.num_heads = n_heads
        else:
            assert in_channels % n_head_channels == 0
            self.num_heads = in_channels // n_head_channels

        self.to_qkv = nn.Conv1d(in_channels, in_channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(nn.Conv1d(in_channels, in_channels, 1))

    def forward(self, x, time=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.to_qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, time=None):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class ResBlock(TimestepBlock):
    def __init__(self, in_channels, time_embed_dim, dropout,
                 out_channels=None, use_conv=False, up=False, down=False, group="none"):
        super().__init__()
        out_channels = out_channels or in_channels

        self.in_layers = nn.Sequential(
            GroupNorm32(32, in_channels),
            nn.SiLU(),
            make_conv2d(in_channels, out_channels, 3, padding=1, group=group),
        )
        self.updown = up or down

        if up:
            self.h_upd = Upsample(in_channels, False, group=group)
            self.x_upd = Upsample(in_channels, False, group=group)
        elif down:
            self.h_upd = Downsample(in_channels, False, group=group)
            self.x_upd = Downsample(in_channels, False, group=group)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.embed_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels),
        )

        self.out_layers = nn.Sequential(
            GroupNorm32(32, out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(make_conv2d(out_channels, out_channels, 3, padding=1, group=group)),
        )

        if out_channels == in_channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = make_conv2d(in_channels, out_channels, 3, padding=1, group=group)
        else:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, time_embed):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.embed_layers(time_embed).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class UNetModel(nn.Module):
    def __init__(self, img_size, base_channels, conv_resample=True,
                 n_heads=1, n_head_channels=-1, channel_mults="",
                 num_res_blocks=2, dropout=0, attention_resolutions="32,16,8",
                 biggan_updown=True, in_channels=1,
                 group="none", use_attention=None):
        self.dtype = torch.float32
        super().__init__()

        if channel_mults == "":
            if img_size == 512:
                channel_mults = (0.5, 1, 1, 2, 2, 4, 4)
            elif img_size == 256:
                channel_mults = (1, 1, 2, 2, 4, 4)
            elif img_size == 128:
                channel_mults = (1, 1, 2, 3, 4)
            elif img_size == 64:
                channel_mults = (1, 2, 3, 4)
            elif img_size == 32:
                channel_mults = (1, 2, 3, 4)
            else:
                raise ValueError(f"unsupported image size: {img_size}")

        if use_attention is None:
            # default off when doing group equivariance
            use_attention = (str(group).strip().lower() in ("none", "no", "0", "false", ""))

        print(f"[UNet] group={group} | use_attention={use_attention} | attention_resolutions={attention_resolutions}")    

        attention_ds = []
        if use_attention and str(attention_resolutions).strip() != "":
            for res in attention_resolutions.split(","):
                if str(res).strip() != "":
                    attention_ds.append(img_size // int(res))

        self.image_size = img_size
        self.in_channels = in_channels
        self.model_channels = base_channels
        self.out_channels = in_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.channel_mult = channel_mults
        self.conv_resample = conv_resample

        self.num_heads = n_heads
        self.num_head_channels = n_head_channels

        time_embed_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            PositionalEmbedding(base_channels, 1),
            nn.Linear(base_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mults[0] * base_channels)
        self.group = group
        self.use_attention = bool(use_attention)

        self.down = nn.ModuleList(
            [TimestepEmbedSequential(make_conv2d(self.in_channels, base_channels, 3, padding=1, group=group))]
        )
        channels = [ch]
        ds = 1

        for i, mult in enumerate(channel_mults):
            for _ in range(num_res_blocks):
                layers = [ResBlock(
                    ch,
                    time_embed_dim=time_embed_dim,
                    out_channels=base_channels * mult,
                    dropout=dropout,
                    group=group,
                )]
                ch = base_channels * mult

                if self.use_attention and ds in attention_ds:
                    layers.append(AttentionBlock(ch, n_heads=n_heads, n_head_channels=n_head_channels))

                self.down.append(TimestepEmbedSequential(*layers))
                channels.append(ch)

            if i != len(channel_mults) - 1:
                out_channels = ch
                self.down.append(TimestepEmbedSequential(
                    ResBlock(ch, time_embed_dim=time_embed_dim, out_channels=out_channels,
                             dropout=dropout, down=True, group=group)
                    if biggan_updown else
                    Downsample(ch, conv_resample, out_channels=out_channels, group=group)
                ))
                ds *= 2
                ch = out_channels
                channels.append(ch)

        middle_layers = [ResBlock(ch, time_embed_dim=time_embed_dim, dropout=dropout, group=group)]
        if self.use_attention:
            middle_layers.append(AttentionBlock(ch, n_heads=n_heads, n_head_channels=n_head_channels))
        middle_layers.append(ResBlock(ch, time_embed_dim=time_embed_dim, dropout=dropout, group=group))
        self.middle = TimestepEmbedSequential(*middle_layers)

        self.up = nn.ModuleList([])

        for i, mult in reversed(list(enumerate(channel_mults))):
            for j in range(num_res_blocks + 1):
                inp_chs = channels.pop()
                layers = [ResBlock(
                    ch + inp_chs,
                    time_embed_dim=time_embed_dim,
                    out_channels=base_channels * mult,
                    dropout=dropout,
                    group=group,
                )]
                ch = base_channels * mult

                if self.use_attention and ds in attention_ds:
                    layers.append(AttentionBlock(ch, n_heads=n_heads, n_head_channels=n_head_channels))

                if i and j == num_res_blocks:
                    out_channels = ch
                    layers.append(
                        ResBlock(ch, time_embed_dim=time_embed_dim,
                                 out_channels=out_channels, dropout=dropout,
                                 up=True, group=group)
                        if biggan_updown else
                        Upsample(ch, conv_resample, out_channels=out_channels, group=group)
                    )
                    ds //= 2

                self.up.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            GroupNorm32(32, ch),
            nn.SiLU(),
            zero_module(make_conv2d(base_channels * channel_mults[0], self.out_channels, 3, padding=1, group=group)),
        )

    def forward(self, x, time):
        time_embed = self.time_embedding(time)

        skips = []
        h = x.type(self.dtype)

        for module in self.down:
            h = module(h, time_embed)
            skips.append(h)

        h = self.middle(h, time_embed)

        for module in self.up:
            h = torch.cat([h, skips.pop()], dim=1)
            h = module(h, time_embed)

        h = h.type(x.dtype)
        return self.out(h)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def update_ema_params(target, source, decay_rate=0.9999):
    targParams = dict(target.named_parameters())
    srcParams = dict(source.named_parameters())
    for k in targParams:
        targParams[k].data.mul_(decay_rate).add_(srcParams[k].data, alpha=1 - decay_rate)