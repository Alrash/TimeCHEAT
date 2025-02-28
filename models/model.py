import math
import numpy as np
import torch
from torch import nn

from models.graph_layer import Encoder
from layers.Transformer_EncDec import Encoder as FormerEncoder, EncoderLayer as FormerEncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class BackBone(nn.Module):
    def __init__(self, channels, attn_head: int = 4, latent_dim: int = 128, n_layers: int = 2,
                 ref_points: int = 128, n_patches: int = 8, dropout: float = 0.1,
                 former_factor: int = 1, former_dff: int = 256, former_output_attention: bool = False,
                 former_layers: int = 3, former_heads: int = 8, former_activation: str = 'gelu',
                 downstream: str = 'classification', config: dict = None):
        super(BackBone, self).__init__()

        self.dim = channels
        self.ath = attn_head
        self.latent_dim = latent_dim
        self.n_layers = n_layers

        self.n_patches = n_patches
        self.register_buffer('patch_range', torch.linspace(0, 1 * config['obs'], self.n_patches + 1))
        assert ref_points % self.n_patches == 0
        self.register_buffer('ref_points', torch.linspace(0, 1 * config['obs'], ref_points))
        self.ref_points = self.ref_points.reshape(self.n_patches, -1)
        # ref_points = torch.linspace(0, 1, ref_points)
        # self.ref_points = []
        # for i in range(self.n_patches):
        #     self.ref_points.append(ref_points[torch.logical_and(ref_points >= self.patch_range[i], ref_points < self.patch_range[i + 1])])
        # self.ref_points[-1] = torch.cat([self.ref_points[-1], torch.tensor([1.])])

        # graph patch
        self.encoder = Encoder(dim=self.dim, attn_head=self.ath, n_patches=self.n_patches, nkernel=self.latent_dim, n_layers=self.n_layers)
        self.position_embedding = PositionalEmbedding(self.ref_points.size(-1))
        self.dropout = nn.Dropout(dropout)

        # transformer
        self.former = FormerEncoder(
            [
                FormerEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, former_factor, attention_dropout=dropout,
                                      output_attention=former_output_attention), self.ref_points.size(-1), former_heads),
                    self.ref_points.size(-1),
                    former_dff,
                    dropout=dropout,
                    activation=former_activation
                ) for _ in range(former_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.ref_points.size(-1))
        )

        self.ds = downstream.lower()
        if self.ds in ['classification']:
            self.flatten = nn.Flatten(start_dim=-2)
            self.pj_dropout = nn.Dropout(dropout)
            self.projection = nn.Linear(self.n_patches * self.ref_points.size(-1) * self.dim, config['n_class'])
        elif self.ds in ['forecast', 'impute']:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(self.n_patches * self.ref_points.size(-1), config['pred_len'])
            self.dropout = nn.Dropout(dropout)
        else:
            raise NotImplementedError

    def _split_patch(self, data, mask, time, i_patch):
        start, end, ref_points = self.patch_range[i_patch], self.patch_range[i_patch + 1], self.ref_points[i_patch].to(data.device)
        time_mask = torch.logical_and(torch.logical_and(time >= start, time <= end), mask.sum(-1) > 0)
        num_observed = time_mask.sum(1).long()

        n_ref_points = ref_points.size(0)
        patch = torch.zeros(data.size(0), num_observed.max() + n_ref_points, self.dim, device=data.device)
        patch_mask, patch_time = torch.zeros_like(patch), torch.zeros(patch.size(0), patch.size(1), device=data.device)
        rp_mask, indices = torch.zeros_like(patch_mask), torch.arange(patch.size(1)).to(data.device)
        for i in range(data.size(0)):
            patch_mask[i, :num_observed[i], :] = mask[i, time_mask[i]]
            patch[i, :num_observed[i], :] = data[i, time_mask[i]]
            patch_time[i, :num_observed[i]] = time[i, time_mask[i]]

            # insert ref points
            patch_time[i, num_observed[i]: num_observed[i] + n_ref_points] = ref_points
            sorted_index = torch.cat([torch.argsort(patch_time[i, :num_observed[i] + n_ref_points]), indices[num_observed[i] + n_ref_points:]])
            patch_mask[i] = patch_mask[i, sorted_index]
            patch[i] = patch[i, sorted_index]
            patch_time[i] = patch_time[i, sorted_index]
            rp_mask[i, sorted_index[num_observed[i]:num_observed[i] + n_ref_points]] = 1.
        # return patch.clone(), patch_mask.clone(), patch_time.clone(), rp_mask.clone()
        return patch, patch_mask, patch_time, rp_mask

    def embedding(self, data):
        vals, mask, time = data[..., :self.dim], data[..., self.dim:-1], data[..., -1]

        # encoder
        repr_patch = []
        for i_patch in range(self.n_patches):
            v, m, t, rp_m = self._split_patch(vals, mask, time, i_patch)

            v = v * m
            context_mask = m + rp_m
            # out -> n_patch * B * {t_i} * channel * latent_dim
            repr, repr_mask, _, _ = self.encoder(t, v, context_mask, rp_m, i_patch)
            # repr_patch.append(repr[repr_mask.sum(-1) > 0, ...].reshape(repr.size(0), -1, self.dim).unsqueeze(1))
            repr_patch.append(repr[repr_mask == 1].reshape(repr.size(0), -1, self.dim).unsqueeze(1))
        # combined, [batch x patch_num x patch_len x dim] => [batch x dim x patch_num x patch_len]
        repr_patch = torch.cat(repr_patch, dim=1).contiguous().permute(0, 3, 1, 2)

        # positional embedding
        repr_patch = torch.reshape(repr_patch, (repr_patch.shape[0] * repr_patch.shape[1], repr_patch.shape[2], repr_patch.shape[3]))
        repr_patch += self.position_embedding(repr_patch)
        repr_patch = self.dropout(repr_patch)

        # transformer encode
        embedding, _ = self.former(repr_patch)
        # [batch x dim x patch_num x patch_len] => [batch x dim x patch_len x patch_num]
        embedding = torch.reshape(embedding, (-1, self.dim, embedding.shape[-2], embedding.shape[-1])).permute(0, 1, 3, 2)
        return embedding

    def forward(self, data, temp=None):
        embedding = self.embedding(data)

        if self.ds in ['classification']:
            out = self.dropout(self.flatten(embedding)).reshape(embedding.shape[0], -1)
            out = self.projection(out)
        elif self.ds in ['forecast', 'impute']:
            out = self.dropout(self.linear(self.flatten(embedding)))
            out = out.permute(0, 2, 1)
        else:
            pass

        return out


if __name__ == '__main__':
    N, T, D = 5, 10, 3
    data = np.arange(N * T * D).reshape(N, T, D)
    mask = np.random.random((N, T)) > 0.5

    select = data[mask, ...]
    select_reshape = select[:int(select.shape[0]//N*N)].reshape(N, -1, D)

    def _split_patch(data, mask, time, i_patch, patch_range, ref_points, dim):
        start, end, ref_points = patch_range[i_patch], patch_range[i_patch + 1], ref_points[i_patch].to(data.device)
        time_mask = torch.logical_and(torch.logical_and(time >= start, time <= end), mask.sum(-1) > 0)
        num_observed = time_mask.sum(1).long()

        n_ref_points = ref_points.size(0)
        patch = torch.zeros(data.size(0), num_observed.max() + n_ref_points, dim, device=data.device)
        patch_mask, patch_time = torch.zeros_like(patch), torch.zeros(patch.size(0), patch.size(1), device=data.device)
        rp_mask, indices = torch.zeros_like(patch_mask), torch.arange(patch.size(1))
        for i in range(data.size(0)):
            patch_mask[i, :num_observed[i], :] = mask[i, time_mask[i]]
            patch[i, :num_observed[i], :] = data[i, time_mask[i]]
            patch_time[i, :num_observed[i]] = time[i, time_mask[i]]

            # insert ref points
            patch_time[i, num_observed[i]: num_observed[i] + n_ref_points] = ref_points
            sorted_index = torch.cat([torch.argsort(patch_time[i, :num_observed[i] + n_ref_points]), indices[num_observed[i] + n_ref_points:]])
            patch_mask[i] = patch_mask[i, sorted_index]
            patch[i] = patch[i, sorted_index]
            patch_time[i] = patch_time[i, sorted_index]
            rp_mask[i, sorted_index[num_observed[i]:num_observed[i] + n_ref_points]] = 1.
        return patch.clone(), patch_mask.clone(), patch_time.clone(), rp_mask.clone()

    N, T, D = 5, 10, 2
    data = torch.randn(N, T, D)
    mask = (torch.randn_like(data) > 0.5).float()
    time = torch.randn(N, T)

    n_patches, ref_points = 4, []
    rps = torch.linspace(0, 1, 128)
    patch_range = torch.linspace(0, 1, n_patches + 1)
    for i in range(n_patches):
        ref_points.append(rps[torch.logical_and(rps >= patch_range[i], rps < patch_range[i + 1])])
    ref_points[-1] = torch.cat([ref_points[-1], torch.tensor([1.])])

    time = (time - torch.min(time, dim=-1, keepdim=True)[0]) / (torch.max(time) - torch.min(time, dim=-1, keepdim=True)[0])
    times = torch.zeros_like(time)
    for i in range(N):
        num_time = torch.randint(2, T, torch.Size([1]))
        t = time[i, :num_time]
        times[i, :num_time] = t[torch.argsort(t)]
        mask[i, num_time:] = torch.zeros(T - num_time, D)

    _split_patch(data, mask, times, 1, patch_range, ref_points, D)
