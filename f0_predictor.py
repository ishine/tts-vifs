import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm

from commons import init_weights, get_padding
from transforms import piecewise_rational_quadratic_transform


class SineGen(nn.Module):
    def __init__(self,
                 sample_rate,
                 harmonic_num=0,
                 sine_amp=0.1,
                 noise_std=0.003,
                 voiced_threshold=0,
                 flag_for_pulse=False,
                 add_noise=False):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = sample_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse
        self.add_noise = add_noise

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        rad_values = (f0_values / self.sampling_rate) % 1
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            tmp_over_one = torch.cumsum(rad_values, 1) % 1
            tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

            sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi)
        else:
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)
            # generate sine waveforms
            sine_waves = self._f02sine(f0_buf) * self.sine_amp
            uv = self._f02uv(f0)
            if self.add_noise:
                noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            else:
                noise_amp = uv * self.noise_std
            noise = noise_amp * torch.randn_like(sine_waves)
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class JointF0(nn.Module):
    def __init__(self, sample_rate, mel_dim=80, f0_step=150):
        super().__init__()
        self.mel_dim = mel_dim
        self.f0_step = f0_step
        # self.wave_gen = SquareGen(sample_rate)
        self.wave_gen = SineGen(sample_rate, flag_for_pulse=True)

    def forward(self, mels, f0s=None):
        pred_log_f0s, pred_vuvs = 0, 0
        f0s = F.interpolate(f0s, scale_factor=self.f0_step, mode='nearest')
        f0s = f0s.transpose(1, 2)  # [B, D=1, T] -> [B, T, D=1]
        wave, _, _ = self.wave_gen(f0s)
        wave = wave.transpose(1, 2)
        # wave: [B, D=1, T]
        # pred_log_f0s: [B, D=1 , T]
        # pred_vuvs: [B, T] -> [B, T]
        return wave, pred_log_f0s, pred_vuvs


class F0Layer(nn.Module):
    def __init__(self, ch0, ch1, ch2, ch3, ch4, down_rates=[3, 4, 5, 5]):
        super().__init__()
        self.f0_layers = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv1d(1, ch4, kernel_size=3, padding=1)),
                nn.LeakyReLU(0.1, inplace=True),
                weight_norm(nn.Conv1d(ch4, ch4, kernel_size=3, padding=1)),
            ),
            nn.Sequential(
                nn.LeakyReLU(0.1, inplace=True),
                weight_norm(nn.Conv1d(ch4, ch4, kernel_size=down_rates[0]*2-1, stride=down_rates[0], padding=down_rates[0]-1)),
                weight_norm(nn.Conv1d(ch4, ch3, kernel_size=3, padding=1)),
                nn.LeakyReLU(0.1, inplace=True),
                weight_norm(nn.Conv1d(ch3, ch3, kernel_size=3, padding=1)),
            ),
            nn.Sequential(
                nn.LeakyReLU(0.1, inplace=True),
                weight_norm(nn.Conv1d(ch3, ch3, kernel_size=down_rates[1]*2-1, stride=down_rates[1], padding=down_rates[1]-1)),
                weight_norm(nn.Conv1d(ch3, ch2, kernel_size=3, padding=1)),
                nn.LeakyReLU(0.1, inplace=True),
                weight_norm(nn.Conv1d(ch2, ch2, kernel_size=3, padding=1)),
            ),
            nn.Sequential(
                nn.LeakyReLU(0.1, inplace=True),
                weight_norm(nn.Conv1d(ch2, ch2, kernel_size=down_rates[2]*2-1, stride=down_rates[2], padding=down_rates[2]-1)),
                weight_norm(nn.Conv1d(ch2, ch1, kernel_size=3, padding=1)),
                nn.LeakyReLU(0.1, inplace=True),
                weight_norm(nn.Conv1d(ch1, ch1, kernel_size=3, padding=1)),
            ),
            nn.Sequential(
                nn.LeakyReLU(0.1, inplace=True),
                weight_norm(nn.Conv1d(ch1, ch1, kernel_size=down_rates[3]*2+1, stride=down_rates[3], padding=down_rates[3])),
                weight_norm(nn.Conv1d(ch1, ch0, kernel_size=3, padding=1)),
                nn.LeakyReLU(0.1, inplace=True),
                weight_norm(nn.Conv1d(ch0, ch0, kernel_size=3, padding=1)),
            ),
        ])

    def forward(self, sines):
        res = []
        for layer in self.f0_layers:
            sines = layer(sines)
            res.append(sines)
        res = res[::-1]
        return res
