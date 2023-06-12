import os
import copy
import random
import math
import commons
import modules
import attentions
import monotonic_align
import utils
import numpy as np
import os.path as osp
import pyworld as pw

import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from f0_predictor import JointF0, F0Layer
from commons import init_weights, get_padding, sequence_mask, rand_slice_segments, generate_path
from torchvision.transforms import ToTensor
from scipy.io.wavfile import read, write
from torchvision.transforms import ToTensor
from scipy.io.wavfile import read, write

import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use('Agg')
# plt.rcParams['font.family'] = 'monospace'
# COLORS = [None, "tomato", "darkviolet", "blue", "green", "red", "cyan", "magenta", "yellow", "black", "white"]
# CMAPS = ['plasma', 'inferno', 'magma', 'cividis']


class TextEncoder(nn.Module):
    def __init__(self,
                 n_vocab,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class Generator(nn.Module):
    def __init__(self,
                 initial_channel,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 use_sine=True,
                 gin_channels=0):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upsamples_rates = upsample_rates
        self.use_sine = use_sine

        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)

        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        chs = []
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            in_channels = upsample_initial_channel // (2 ** i)
            out_channels = upsample_initial_channel // (2 ** (i + 1))
            if i == 0:
                chs.append(in_channels)
            chs.append(out_channels)
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(in_channels, out_channels, k, u, padding=(k - u) // 2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        if self.use_sine:
            self.joint_f0 = JointF0(24000, mel_dim=chs[0], f0_step=150)
            self.f0_layer = F0Layer(chs[0], chs[1], chs[2], chs[3], chs[4], [3, 4, 5, 5])

    def forward(self, x, g=None, f0s=None):
        x = self.conv_pre(x)
        if self.use_sine:
            wave, pred_log_f0s, pred_vuvs = self.joint_f0(x, f0s)
            f0_conds = self.f0_layer(wave)
        else:
            pred_log_f0s, pred_vuvs = 0, 0

        if g is not None:
            x = x + self.cond(g)
        if self.use_sine:
            x = x + f0_conds[0]

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            if self.use_sine:
                x = x + f0_conds[i + 1]
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x, pred_log_f0s, pred_vuvs


class PosteriorEncoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(self,
                channels,
                hidden_channels,
                kernel_size,
                dilation_rate,
                n_layers,
                n_flows=4,
                affine=False,
                gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.affine = affine

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(modules.ResidualCouplingLayer(channels,
                                                            hidden_channels,
                                                            kernel_size,
                                                            dilation_rate,
                                                            n_layers,
                                                            gin_channels=gin_channels,
                                                            mean_only=True if not affine else False))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        logdet_tot = 0.0
        if not reverse:
            for flow in self.flows:
                x, logdet = flow(x, x_mask, g=g, reverse=reverse)
                logdet_tot += logdet
        else:
            for flow in reverse(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x, logdet_tot


class ResidualCouplingBlock_Condition():
    def __init__(self,
                 channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 n_flows=4,
                 affine=False,
                 gin_channels=0):

        super(ResidualCouplingBlock_Condition, self).__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.affine = affine


        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(modules.ResidualCouplingLayer_Condition(channels, hidden_channels, kernel_size,
                                                                        dilation_rate,
                                                                        n_layers,
                                                                        gin_channels=gin_channels,
                                                                        mean_only=True if not affine else False))
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, c, g=None, reverse=False):
        logdet_tot = 0.0
        if not reverse:
            for flow in self.flows:
                x, logdet = flow(x, x_mask, c, g=g, reverse=reverse)
                logdet_tot += logdet
        else:
            for flow in reverse(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x, logdet_tot


class StochasticDurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        
        for i in range(n_flows):
            self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, random_input=None, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)

        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
            logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1, 2]) - logdet_tot_q

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1, 2]) - logdet_tot
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            if random_input is None:
                z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
            else:
                z = random_input[None, None, : x.size(2)].expand(x.size(0), 2, -1).to(device=x.device, dtype=x.dtype) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask



class VITS(nn.Module):
    def __init__(self,
                 n_vocab,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 n_speakers=0,
                 gin_channels=0,
                 use_sdp=True,
                 **kwargs):

        super(VITS, self).__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_sdp = use_sdp

        assert self.inter_channels == self.hidden_channels

        self.enc_p = TextEncoder(n_vocab,
                                 inter_channels,
                                 hidden_channels,
                                 filter_channels,
                                 n_heads,
                                 n_layers,
                                 kernel_size,
                                 p_dropout)
        

        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        if use_sdp:
            self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
        else:
            self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)



class VIFSpeechV1(nn.Module):
    def __init__(self,
                 n_vocab,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 n_speakers=0,
                 gin_channels=0,
                 use_sdp=True,
                 use_mas=True,
                 **kwargs):

        super(VIFSpeech, self).__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_sdp = use_sdp
        self.use_mas = use_mas

        assert self.inter_channels == self.hidden_channels

        self.enc_p = TextEncoder(n_vocab,
                                 inter_channels,
                                 hidden_channels,
                                 filter_channels,
                                 n_heads,
                                 n_layers,
                                 kernel_size,
                                 p_dropout)

        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

        if use_sdp:
            self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels)
        else:
            self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)


    def forward(self, **kwargs):
        x = kwargs['x']
        y = kwargs['spec']
        x_lengths = kwargs['x_lengths']
        y_lengths = kwargs['spec_lengths']
        sid = kwargs['speaker_ids']

        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        
        if self.use_mas:
            with torch.no_grad():
                # negative cross-entropy
                s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
                neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)  # [b, 1, t_s]
                neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r)  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
                neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r))  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
                neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True)  # [b, 1, t_s]
                neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

                attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
                attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()
            w = attn.sum(2)
        else:
            w = kwargs['durations']
            y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = generate_path(w, attn_mask).float().detach()
            w = w.unsqueeze(dim=1).half()

        if self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_)**2, [1, 2]) / torch.sum(x_mask)  # for averaging

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)
        z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)
        return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)


    def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None
        if self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)


class TextEncoder_NoProb(nn.Module):
    def __init__(self,
                 n_vocab,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super(TextEncoder_NoProb, self).__init__()
        self.n_vocab = n_vocab
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout)

        self.pt2_upsampler = None
        self.mel_generator = None

    def forward(self, x, x_lengths):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)
        x = x * x_mask
        return x, x_mask


class ProsodyPredictor_Transformer_KL(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 filter_channels,
                 gin_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout):
        super(ProsodyPredictor_Transformer_KL, self).__init__()
        self.out_channels = out_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.cond = nn.Conv1d(gin_channels, hidden_channels, 1)
        self.encoder = attentions.Encoder(hidden_channels,
                                          filter_channels,
                                          n_heads,
                                          n_layers,
                                          kernel_size,
                                          p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_mask, g=None, noise_scale=1.0):
        x = self.pre(x)
        if g is not None:
            x = x + self.cond(g)
        x = x * x_mask
        x = self.encoder(x, x_mask)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs) * noise_scale)

        return z, m, logs


class DurationPredictor_prosody(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
        super(DurationPredictor_prosody, self).__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)
        self.relu = nn.ReLU()

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        self.pre_1 = nn.Conv1d(in_channels, in_channels, 1)
        self.pre_2 = nn.Conv1d(in_channels, in_channels, 1)

    def forward(self, x, prosody_hidden, x_mask, g=None):
        x = self.pre_1(x) + self.pre_2(prosody_hidden)
        if g is not None:
            x = x + self.cond(g)

        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)

        x = self.relu(x)
        return x * x_mask


class ConditionFusionLayer_WN_Add(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 spec_channels,
                 gin_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers):
        super(ConditionFusionLayer_WN_Add, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.spec_channels = spec_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.cond = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels,
                              kernel_size,
                              dilation_rate,
                              n_layers,
                              gin_channels=gin_channels)
        self.post_mel = nn.Conv1d(hidden_channels, 80, 1)

    def forward(self, x, c, x_mask, g=None):
        x = self.pre(x)
        c = self.cond(c)
        x = x + c
        x = x * x_mask
        x = self.enc(x, x_mask, g=g)
        mel = self.post_mel(x) * x_mask
        return x, mel


class F0PredictorV2(nn.ModuleList):
    def __init__(self, ir_dim=192, gin_channels=256):
        super().__init__()
        self.cond = nn.Conv1d(gin_channels, ir_dim, 1)
        self.f0_predictor = nn.Sequential(
            nn.Conv1d(ir_dim, 128, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(128, 64, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(64, 64, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.f0_logit_layer = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )
        self.f0_value_layer = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )

    def forward(self, x, g):
        x = x + self.cond(g)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.f0_predictor(x)
        pred_log_f0s = self.f0_value_layer(x).clamp(0)
        pred_f0s = torch.exp(pred_log_f0s) - 1.0
        pred_vuvs = self.f0_logit_layer(x)
        return pred_log_f0s, pred_vuvs.squeeze(1), pred_f0s.clamp(0)


class Generator_Original(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator_Original, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.upsamples_rates = upsample_rates

        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)

        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class ConditionPosteriorEncoder_PhoneLevel(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 gin_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers):
        super(ConditionPosteriorEncoder_PhoneLevel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.cond = nn.Conv1d(hidden_channels, hidden_channels, 1)
        self.encoder = modules.WN(hidden_channels,
                                  kernel_size,
                                  dilation_rate,
                                  n_layers,
                                  gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, c, x_lengths, attention, durations, phone_mask, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        c = self.cond(c) * x_mask
        x = x + c
        x = self.encoder(x, x_mask, g=g)

        attention_for_phone = attention.contiguous().transpose(1, 2)
        x = torch.bmm(x, attention_for_phone) * phone_mask
        durations = 1.0 / (durations + 1e-6)
        x = x * durations
        prosody_hidden = torch.detach(x)

        stats = self.proj(x) * phone_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs))

        return z, m, logs, prosody_hidden, x_mask

    def infer(self, prosody_hidden, phone_mask):
        stats = self.proj(prosody_hidden) * phone_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs))
        return z


class PosteriorWaveEncoder(nn.Module):
    def __init__(self,
                 spec_channels,
                 hidden_channels,
                 kernel_size,
                 dilation_rate,
                 n_layers,
                 gin_channels=0):
        super(PosteriorWaveEncoder, self).__init__()
        self.spec_channels = spec_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers

        self.pre = nn.Conv1d(spec_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels,
                              kernel_size,
                              dilation_rate,
                              n_layers,
                              gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, hidden_channels * 2, 1)

    def forward(self, x, x_mask, g=None):
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.hidden_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs))
        return z, m, logs


class VIFSpeechV2(nn.Module):
    def __init__(self,
                 embedding_dict,
                 combined_mode,
                 padding_idx,
                 normalize,
                 n_vocab,
                 spec_channels,
                 segment_size,
                 inter_channels,
                 hidden_channels,
                 filter_channels,
                 n_heads,
                 n_layers,
                 kernel_size,
                 p_dropout,
                 resblock,
                 resblock_kernel_sizes,
                 resblock_dilation_sizes,
                 upsample_rates,
                 upsample_initial_channel,
                 upsample_kernel_sizes,
                 n_speakers=0,
                 gin_channels=0,
                 use_tacolabel=True,
                 affine=False,
                 sdp=False,
                 finetune=False,
                 speaker_id=-1,
                 infer=False,
                 f0_min=42.0,
                 f0_max=1000.0,
                 **kwargs):

        super(VIFSpeechV2, self).__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_tacolabel = use_tacolabel
        self.affine = affine
        self.sdp = sdp

        assert self.inter_channels == self.hidden_channels

        self.text_enc = TextEncoder_NoProb(n_vocab,
                               hidden_channels,
                               filter_channels,
                               n_heads,
                               n_layers,
                               kernel_size,
                               p_dropout)

        # Prosody Predictor
        self.prosody_predictor = ProsodyPredictor_Transformer_KL(in_channels=hidden_channels,
                                                                 out_channels=hidden_channels,
                                                                 hidden_channels=hidden_channels,
                                                                 filter_channels=filter_channels,
                                                                 gin_channels=gin_channels,
                                                                 n_heads=n_heads,
                                                                 n_layers=n_layers,
                                                                 kernel_size=kernel_size,
                                                                 p_dropout=p_dropout)

        self.duration_predictor = DurationPredictor_prosody(hidden_channels, 256, 5, 0.5)

        self.acoustic_encoder = ConditionFusionLayer_WN_Add(in_channels=hidden_channels,
                                                            hidden_channels=hidden_channels,
                                                            spec_channels=spec_channels,
                                                            gin_channels=gin_channels,
                                                            kernel_size=5,
                                                            dilation_rate=1,
                                                            n_layers=8)

        self.wave_decoder = Generator(hidden_channels,
                                      resblock,
                                      resblock_kernel_sizes,
                                      resblock_dilation_sizes,
                                      upsample_rates,
                                      upsample_initial_channel,
                                      upsample_kernel_sizes,
                                      use_sine=True,
                                      gin_channels=gin_channels)

        self.f0_min = f0_min
        self.f0_max = f0_max

        self.f0_predictor = F0PredictorV2(ir_dim=hidden_channels, gin_channels=gin_channels)

        self.prosody_enhancer = Generator_Original(initial_channel=hidden_channels,
                                                   resblock="2",
                                                   resblock_kernel_sizes=[3, 5, 7],
                                                   resblock_dilation_sizes=[[1, 2], [2, 6], [3, 12]],
                                                   upsample_rates=[10, 6, 5],
                                                   upsample_initial_channel=256,
                                                   upsample_kernel_sizes=[20, 12, 11],
                                                   gin_channels=gin_channels)

        self.posterior_prosody_encoder = ConditionPosteriorEncoder_PhoneLevel(spec_channels,
                                                                              inter_channels,
                                                                              hidden_channels,
                                                                              gin_channels,
                                                                              5, 1, 8)

        self.prosody_flow = ResidualCouplingBlock(channels=inter_channels,
                                                  hidden_channels=hidden_channels,
                                                  kernel_size=5,
                                                  dilation_rate=1,
                                                  n_layers=4,
                                                  n_flows=4,
                                                  affine=affine,
                                                  gin_channels=gin_channels)

        self.wave_flow = ResidualCouplingBlock_Condition(channels=inter_channels,
                                                         hidden_channels=hidden_channels,
                                                         kernel_size=5,
                                                         dilation_rate=1,
                                                         n_layers=4,
                                                         n_flows=4,
                                                         affine=affine,
                                                         gin_channels=gin_channels)

        self.posterior_wave_encoder = PosteriorWaveEncoder(spec_channels=spec_channels,
                                                           hidden_channels=hidden_channels,
                                                           kernel_size=5,
                                                           dilation_rate=1,
                                                           n_layers=8,
                                                           gin_channels=gin_channels)

        self.finetune = finetune
        if self.finetune:
            print("[INFO]: Frozen text encoder...")
            for param in self.text_enc.parameters():
                param.requires_grad = False
            self.speaker_id = speaker_id

        if n_speakers > 1:
            self.emb_speaker = nn.Embedding(n_speakers, gin_channels)

    def calculate_f0(self, wav, sample_rate, f0_ms=6.25, f0_floor=42.0, f0_ceil=1000.0, mode='harvest'):
        f0_fn = pw.harvest if mode == 'harvest' else pw.dio
        f0, t = f0_fn(wav.astype(np.float64),
                      sample_rate,
                      frame_period=f0_ms,
                      f0_floor=f0_floor,
                      f0_ceil=f0_ceil)
        f0 = pw.stonemask(wav.astype(np.float64), f0, t, sample_rate)
        return f0

    def forward(self, **kwargs):
        y = kwargs['spec']
        y_lengths = kwargs['spec_lengths']
        sid = kwargs['speaker_ids']
        if self.finetune and self.speaker_id != -1:
            sid[:] = self.speaker_id
            sid = sid.long()

        # Encode text
        if self.use_tacolabel:
            x, x_mask = self.text_enc(**kwargs)
        else:
            x = kwargs["x"]
            x_lengths = kwargs["x_lengths"]
            x, x_mask = self.text_enc(x, x_lengths)

        if self.n_speakers > 0:
            g = self.emb_speaker(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        # Expand linguistic feature
        w = kwargs["durations"]
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w, attn_mask).float().detach().contiguous().squeeze(1).transpose(1, 2)
        w = w.unsqueeze(dim=1).float()

        # Expand
        x_expand = torch.bmm(x, attn)

        # Condition
        z, m_q, logs_q, prosody_hidden, y_mask = self.posterior_prosody_encoder(y, x_expand, y_lengths, attn, w, x_mask, g=g)

        # Predict prosody
        z_hat, m_pp, logs_pp = self.prosody_predictor(x.detach(), x_mask, g=g.detach())

        # Predict duration
        w_hat = self.duration_predictor(x, z, x_mask)
        l_length = F.l1_loss(w, w_hat)

        # Flow to normal distribution
        z_p, logdet = self.prosody_flow(z, x_mask, g=g)

        # Expand z
        z = torch.bmm(z, attn) * y_mask

        # Random mask x_expand
        random_mask = torch.ones_like(x_expand)[:, :, :1]
        random_index = random.choices([i for i in range(random_mask.size(1))], k=int(random_mask.size(1) * 0.5))
        random_mask[:, random_index, :] = 0.
        x_expand = x_expand * random_mask

        # Fuse condition with z
        intermediate_representation, z_mel_hat = self.acoustic_encoder(x_expand, z, y_mask, g=g)

        # Wave Flow
        z_pwe, m_pwe, logs_pwe = self.posterior_wave_encoder(y, y_mask, g=g)
        z_wf, logdet_wf = self.wave_flow(z_pwe, y_mask, intermediate_representation, g=g)

        # Random cut
        z_slice, ids_slice = commons.rand_slice_segments(z_pwe, y_lengths, self.segment_size)

        # Prosody enhanced
        ir_slice = commons.slice_segments(intermediate_representation, ids_slice, self.segment_size)
        o_aux = self.prosody_enhancer(ir_slice, g=g)

        # Generator
        bce_loss = nn.BCEWithLogitsLoss()
        f0s = commons.slice_segments(kwargs['f0'].unsqueeze(1), ids_slice*2, self.segment_size*2)

        o, _, _ = self.wave_decoder(z_slice, g=g, f0s=f0s)
        pred_log_f0s, pred_vuvs, pred_f0s = self.f0_predictor(intermediate_representation.detach(), g=g.detach())

        f0_mask = torch.unsqueeze(commons.sequence_mask(y_lengths * 2, None), 1).to(x_mask.dtype)
        vuv_targets = (kwargs['f0'] * f0_mask.squeeze(1) > 0).float()
        l_vuv = bce_loss(pred_vuvs * f0_mask.squeeze(1), vuv_targets)
        l_f0 = F.l1_loss(pred_log_f0s * f0_mask * vuv_targets.unsqueeze(1), torch.log(kwargs['f0'].unsqueeze(1) + 1) * f0_mask)

        return o, o_aux, ids_slice, x_mask, y_mask, z_mel_hat, l_vuv, l_f0, (z_wf, logdet_wf, m_pwe, logs_pwe), (z_p, logdet, m_q, logs_q, m_pp, logs_pp), l_length

    def infer(self, data_dict, sid=None, sid_pp=None, ones=0, test_vocoder=False, noise_scale=1.0, length_scale=1.0, max_len=None, use_gt_duration=False, reference=None, use_prosody_predictor=False, ifcuda=False, ref_audio=None, CAL_F0_FROM_WAV=False):
        if sid is None:
            sid = data_dict['speaker_ids']
        else:
            if ifcuda:
                sid = torch.Tensor([sid]).long().cuda()
            else:
                sid = torch.Tensor([sid]).long()

        if sid_pp is not None:
            if ifcuda:
                sid_pp = torch.Tensor([sid_pp]).long().cuda()
            else:
                sid_pp = torch.Tensor([sid_pp]).long()

        # Encode text
        if self.use_tacolabel:
            x, x_mask = self.text_enc(**data_dict)
        else:
            x = data_dict["x"]
            x_lengths = data_dict["x_lengths"]
            x, x_mask = self.text_enc(x, x_lengths)

        if self.n_speakers > 0:
            g = self.emb_speaker(sid).unsqueeze(-1)  # [b, h, 1]
            if sid_pp is not None:
                g_pp = self.emb_speaker(sid_pp).unsqueeze(-1)
            else:
                g_pp = g
        else:
            g = None

        # Prosody predictor
        z, _, _ = self.prosody_predictor(x, x_mask, g=g_pp, noise_scale=noise_scale)

        # Predict duration
        w = self.duration_predictor(x, z, x_mask)

        if use_gt_duration:
            w = data_dict["durations"].unsqueeze(1).float() * x_mask * length_scale
        else:
            w = w * x_mask * length_scale
        w_ceil = torch.round(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask).float().contiguous().squeeze(1).transpose(1, 2)

        # Expand
        x = torch.bmm(x, attn)

        if not use_prosody_predictor:
            if reference is None:
                # Flow to normal distribution
                if ones == 0:
                    z_p = 0.0 + torch.zeros_like(x_mask.expand(-1, x.size(1), -1)) * 1.0 * noise_scale
                elif ones == 1:
                    z_p = 0.0 + torch.ones_like(x_mask.expand(-1, x.size(1), -1)) * 1.0 * noise_scale
                elif ones == -1:
                    z_p = 0.0 + torch.ones_like(x_mask.expand(-1, x.size(1), -1)) * -1.0 * 1.0 * noise_scale
                elif ones == None:
                    z_p = ((torch.randn_like(x_mask) > 0.0).float() * 2.0 - 1.0).expand(-1, x.size(1), -1)
                else:
                    raise Exception("Unsupport ones!")
                z, _ = self.prosody_flow(z_p, x_mask, g=g, reverse=True)
            else:
                if test_vocoder:
                    f0s = self.calculate_f0(ref_audio[:, :y_lengths[0] * 300][0].numpy(), 24000)
                    f0s = torch.from_numpy(f0s).unsqueeze(0).unsqueeze(0).float()
                    reference = reference[:, :, :y_lengths[0]]  # one batch
                    z_pwe, m_pwe, logs_pwe = self.posterior_wave_encoder(reference, y_mask, g=g)

                    if f0s.size(-1) % 2 != 0:
                        f0s = f0s[:, :, :-1]
                    if f0s.size(-1) > z_pwe.size(-1) * 2:
                        f0s = f0s[:, :, :z_pwe.size(-1) * 2]
                    elif f0s.size(-1) < z_pwe.size(-1) * 2:
                        z_pwe = z_pwe[:, :, :f0s.size(-1) // 2]

                    o, _, _ = self.wave_decoder(z_pwe, g=g, f0s=f0s)
                    return o, x_mask, y_mask, (None, None)
                else:
                    reference = reference[:, :, :y_lengths[0]]  # one batch
                    z, _, _, _, _ = self.posterior_prosody_encoder(reference, x, y_lengths, attn, w_ceil, x_mask, g=g)

        z = torch.bmm(z, attn) * y_mask

        # Fuse condition with z
        intermediate_representation, mel = self.acoustic_encoder(x, z, y_mask, g=g)
        z_from_flow, _ = self.wave_flow(torch.randn_like(intermediate_representation), y_mask, intermediate_representation, g=g, reverse=True)

        # Predict f0s
        pred_log_f0s, pred_vuvs, pred_f0s = self.f0_predictor(intermediate_representation, g=g)
        pred_f0s = pred_f0s.clamp(self.f0_min, self.f0_max)
        f0s = pred_f0s * (pred_vuvs.unsqueeze(1) > -0.5).float()  # threshold

        # Cut max length
        if CAL_F0_FROM_WAV:
            o_from_ir = self.prosody_enhancer(intermediate_representation, g=g)
            f0s = self.calculate_f0(o_from_ir[:, 0, :][:, :y_lengths[0] * 300][0].numpy(), 24000)
            f0s = torch.from_numpy(f0s).unsqueeze(0).unsqueeze(0).float()[:, :, :-1]
        o, _, _ = self.wave_decoder(z_from_flow * y_mask, g=g, f0s=f0s)

        return o, x_mask, y_mask, (z, mel)


