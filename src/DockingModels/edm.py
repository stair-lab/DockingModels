import math
import torch
import numpy as np
from torch import nn
from DockingModels.registry import register_model
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter, scatter_mean
from DockingModels.egnn import EGNN, EquivariantBlock
from transformers import PreTrainedModel, PretrainedConfig
import copy

def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x

class GELUMLP(nn.Module):
    """Simple MLP with post-LayerNorm"""

    def __init__(
        self,
        n_in_feats,
        n_out_feats,
        n_hidden_feats=None,
        dropout=0.0,
        zero_init=False,
    ):
        super(GELUMLP, self).__init__()
        self.dropout = dropout
        if n_hidden_feats is None:
            self.layers = nn.Sequential(
                nn.Linear(n_in_feats, n_in_feats),
                nn.GELU(),
                nn.LayerNorm(n_in_feats),
                nn.Linear(n_in_feats, n_out_feats),
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(n_in_feats, n_hidden_feats),
                nn.GELU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(n_hidden_feats, n_hidden_feats),
                nn.GELU(),
                nn.LayerNorm(n_hidden_feats),
                nn.Linear(n_hidden_feats, n_out_feats),
            )
        nn.init.xavier_uniform_(self.layers[0].weight, gain=1)
        # zero init for residual branches
        if zero_init:
            self.layers[-1].weight.data.fill_(0.0)
        else:
            nn.init.xavier_uniform_(self.layers[-1].weight, gain=1)

    def _zero_init(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers(x)

class GaussianSmearing(torch.nn.Module):
    # used to embed the edge distances
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

class FourierEmbedding(nn.Module):
    """ Algorithm 22 """

    def __init__(self, dim, sigma_data=16, eps = 1e-20):
        super().__init__()
        self.proj = nn.Linear(1, dim)
        self.proj.requires_grad_(False)
        self.norm_fourier = nn.LayerNorm(dim)
        self.fourier_to_single = nn.Linear(dim, dim, bias = False)
        self.eps = eps
        self.sigma_data = sigma_data

    def forward(self, times):
        rand_proj = self.proj(times)
        n = torch.cos(2 * math.pi * rand_proj)
        n = self.norm_fourier(n)
        n = self.fourier_to_single(n)
        return rand_proj

class AtomEncoder(nn.Module):
    def __init__(self, emb_dim, feature_dims, sigma_embed_dim, lm_embedding_type=None, h_dim=None):
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1] + sigma_embed_dim
        self.lm_embedding_type = lm_embedding_type

        # Linear layer for positional encoding of first (N, 3) elements
        self.pos_linear = nn.Linear(3, emb_dim)
        self.feature_linear = nn.Linear(self.num_categorical_features, emb_dim)

        self.time_linear = nn.Linear(sigma_embed_dim, emb_dim)

        self.lm_embedding_dim = 0
        if self.lm_embedding_type is not None:
            if self.lm_embedding_type == 'esm':
                self.lm_embedding_dim = 1280
            else:
                raise ValueError('LM Embedding type was not correctly determined. LM embedding type:', self.lm_embedding_type)
            self.lm_embedding_layer = nn.Linear(self.lm_embedding_dim, emb_dim)

        h_dim = emb_dim if h_dim is None else h_dim
        input_dim = emb_dim * 3 if self.lm_embedding_type is None else emb_dim * 4
        self.embed = nn.Sequential(nn.Linear(input_dim, h_dim), nn.ReLU(), nn.Linear(h_dim, h_dim))

    def forward(self, x, dtype=torch.float32):
        if self.lm_embedding_type is not None:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features + self.lm_embedding_dim + 3
        else:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features + 3

        pos_embedding = self.pos_linear(x[:, :3])

        feature_embedding = self.feature_linear(x[:, 3:3+self.num_categorical_features])

        time_embedding = self.time_linear(x[:, -self.num_scalar_features:])

        if self.lm_embedding_type is not None:
            lm_embedding = self.lm_embedding_layer(x[:, 3+self.num_categorical_features:self.lm_embedding_dim+3+self.num_categorical_features])

        x_embedding = torch.cat([pos_embedding, feature_embedding, lm_embedding, time_embedding], axis=1) if self.lm_embedding_type is not None else torch.cat([pos_embedding, feature_embedding, time_embedding], axis=1)
        x_embedding = self.embed(x_embedding)
        return x_embedding

class EGNNModel(nn.Module):
    def __init__(self, device, in_lig_edge_features=4, sigma_embed_dim=32, in_dim=8,
                 in_node_nf=16, num_layers=2, lig_max_radius=5, rec_max_radius=30,
                 cross_max_distance=250, distance_embed_dim=32, cross_distance_embed_dim=32,
                 batch_norm=True, dropout=0.0, lm_embedding_type=None):

        super(EGNNModel, self).__init__()
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.in_node_nf = in_node_nf
        self.device = device

        self.in_dim = in_dim

        self.time_embed = FourierEmbedding(sigma_embed_dim)
        lig_feature_dims = ([119, 4, 12, 12, 8, 10, 6, 6, 2, 8, 2, 2, 2, 2, 2, 2], 0)
        rec_residue_feature_dims = ([38], 0)

        self.num_layers = num_layers
        self.lig_node_embedding = AtomEncoder(emb_dim=in_dim, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim, h_dim=in_node_nf)
        self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim, in_node_nf),nn.ReLU(), nn.Dropout(dropout),nn.Linear(in_node_nf, in_node_nf))

        self.rec_node_embedding = AtomEncoder(emb_dim=in_dim, feature_dims=rec_residue_feature_dims, sigma_embed_dim=sigma_embed_dim, lm_embedding_type=lm_embedding_type, h_dim=in_node_nf)
        self.rec_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, in_node_nf), nn.ReLU(), nn.Dropout(dropout),nn.Linear(in_node_nf, in_node_nf))

        self.cross_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, in_node_nf), nn.ReLU(), nn.Dropout(dropout),nn.Linear(in_node_nf, in_node_nf))
        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)

        self.emb_layers = 3
        for i in range(self.emb_layers):
            parameters = {
                'hidden_nf': in_node_nf,
                'edge_feat_nf': 1 + in_node_nf*3,
                'device': device,
                'tanh': False,
                'n_layers': 2,

            }
            self.add_module("lig_e_block_%d" % i, EquivariantBlock(**parameters, normalization_factor=1000))
            self.add_module("rec_e_block_%d" % i, EquivariantBlock(**parameters, normalization_factor=9000))

        for i in range(num_layers):
            parameters = {
                'hidden_nf': in_node_nf,
                'edge_feat_nf': 1 + in_node_nf*3,
                'device': device,
                'tanh': False,
                'n_layers': 2,

            }
            self.add_module("e_block_%d" % i, EquivariantBlock(**parameters, normalization_factor=10000))

        self.linear = GELUMLP(3,3,32)


    def forward(self, data, times, dtype):

        # build ligand graph
        lig_node_attr, lig_edge_index, lig_edge_attr = self.build_lig_conv_graph(data, times)
        lig_src, lig_dst = lig_edge_index
        lig_node_attr = self.lig_node_embedding(lig_node_attr, dtype=dtype)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)
        lig_coords = data['ligand'].pos.clone()

        # build receptor graph
        rec_node_attr, rec_edge_index, rec_edge_attr = self.build_rec_conv_graph(data, times)
        rec_src, rec_dst = rec_edge_index
        rec_node_attr = self.rec_node_embedding(rec_node_attr, dtype=dtype)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)
        rec_coords = data['receptor'].pos.clone()

        for l in range(self.emb_layers):
            lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_edge_index[0], :self.in_node_nf], lig_node_attr[lig_edge_index[1], :self.in_node_nf]], -1)
            rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_edge_index[0], :self.in_node_nf], rec_node_attr[rec_edge_index[1], :self.in_node_nf]], -1)

            lig_node_attr, lig_coords = self._modules["lig_e_block_%d" % l](lig_node_attr, lig_coords, lig_edge_index, edge_attr=lig_edge_attr_, dtype=dtype)
            rec_node_attr, rec_coords = self._modules["rec_e_block_%d" % l](rec_node_attr, rec_coords, rec_edge_index, edge_attr=rec_edge_attr_, dtype=dtype)

        # build cross graph
        cross_edge_index, cross_edge_attr = self.build_cross_conv_graph(data, self.cross_max_distance)
        cross_lig, cross_rec = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)

        coords = torch.cat([lig_coords, rec_coords], dim=0)
        node_attr = torch.cat([lig_node_attr, rec_node_attr], dim=0)
        cross_edge_index[1] = cross_edge_index[1] + len(lig_node_attr)
        edge_index = torch.cat([lig_edge_index, cross_edge_index, rec_edge_index + len(lig_node_attr),
                                torch.flip(cross_edge_index, dims=[0])], dim=1)
        edge_attr = torch.cat([lig_edge_attr, cross_edge_attr, rec_edge_attr, cross_edge_attr], dim=0)

        for i in range(self.num_layers):
            edge_attr_ = torch.cat([edge_attr, node_attr[edge_index[0], :self.in_node_nf], node_attr[edge_index[1], :self.in_node_nf]], -1)
            node_attr, coords = self._modules["e_block_%d" % i](node_attr, coords, edge_index, edge_attr=edge_attr_, dtype=dtype)

        lig_coords = coords[:len(lig_coords)]
        vel = lig_coords
        vel = self.linear(vel)
        return vel

    def build_lig_conv_graph(self, data, times):
        # builds the ligand graph edges and initial node and edge features
        t_embed = self.time_embed(times)
        data['ligand'].node_sigma_emb = t_embed[data['ligand'].batch]
        # compute edges
        radius_edges = radius_graph(data['ligand'].pos, self.lig_max_radius, data['ligand'].batch)
        edge_index = torch.cat([data['ligand', 'ligand'].edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data['ligand', 'ligand'].edge_attr,
            torch.zeros(radius_edges.shape[-1], self.in_lig_edge_features, device=data['ligand'].x.device)
        ], 0)

        # compute initial features
        edge_sigma_emb = data['ligand'].node_sigma_emb[edge_index[0].long()]

        node_attr = torch.cat([data['ligand'].pos, data['ligand'].x, data['ligand'].node_sigma_emb], 1)

        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = torch.cat([edge_attr, edge_sigma_emb, edge_length_emb], 1)

        return node_attr, edge_index, edge_attr

    def build_rec_conv_graph(self, data, times):
        # builds the receptor initial node and edge embeddings

        t_embed = self.time_embed(times)
        data['receptor'].node_sigma_emb = t_embed[data['receptor'].batch]

        node_x = torch.cat([data['receptor'].x, data['receptor'].lm_embeddings], 1)
        node_attr = torch.cat([data['receptor'].pos, node_x, data['receptor'].node_sigma_emb], 1)

        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data['receptor', 'receptor'].edge_index
        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['receptor'].pos[src.long()]

        edge_length_emb = self.rec_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['receptor'].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)

        return node_attr, edge_index, edge_attr

    def build_cross_conv_graph(self, data, cross_distance_cutoff):
        # builds the cross edges between ligand and receptor
        if torch.is_tensor(cross_distance_cutoff):
            # different cutoff for every graph (depends on the diffusion time)
            edge_index = radius(data['receptor'].pos / cross_distance_cutoff[data['receptor'].batch],
                                data['ligand'].pos / cross_distance_cutoff[data['ligand'].batch], 1,
                                data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)
        else:
            edge_index = radius(data['receptor'].pos, data['ligand'].pos, cross_distance_cutoff,
                            data['receptor'].batch, data['ligand'].batch, max_num_neighbors=10000)

        src, dst = edge_index
        edge_vec = data['receptor'].pos[dst.long()] - data['ligand'].pos[src.long()]

        edge_length_emb = self.cross_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data['ligand'].node_sigma_emb[src.long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)

        return edge_index, edge_attr


class CustomConfig(PretrainedConfig):
    model_type = "equivariant_elucidated_diffusion"

    def __init__(
        self,
        sigma_data=32,       # standard deviation of data distribution
        sigma_min=0.002,     # min noise level
        sigma_max=160,         # max noise level
        rho=7,
        S_churn=40,
        S_min=0.05,
        S_max=50,
        S_noise=1.003,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.S_churn = S_churn
        self.S_min = S_min
        self.S_max = S_max
        self.S_noise = S_noise


class EquivariantElucidatedDiffusion(PreTrainedModel):
    config_class = CustomConfig

    def __init__(
        self,
        config: CustomConfig,
        net: EGNNModel,
    ):
        super().__init__(config)
        self.net = net
        self.sigma_min = config.sigma_min
        self.sigma_max = config.sigma_max
        self.sigma_data = config.sigma_data
        self.rho = config.rho
        self.S_churn = config.S_churn
        self.S_min = config.S_min
        self.S_max = config.S_max
        self.S_noise = config.S_noise

    @torch.no_grad()
    def sample(self, data, num_steps, dtype=torch.float32):
        x_latents = []
        batch = data['ligand'].batch
        batch_size = torch.unique(batch).shape[0]
        step_indices = torch.arange(num_steps, device=self.device)
        t_steps = self.sigma_data * ((self.sigma_max ** (1 / self.rho) + step_indices / (num_steps - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))) ** self.rho)
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

        data.atom_pos_ground_truth = copy.deepcopy(data['ligand'].pos)
        x_next = t_steps[0] * torch.randn_like(data['ligand'].pos, device = self.device)
        x_latents.append(x_next)

        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            gamma = min(self.S_churn / num_steps, np.sqrt(2) - 1) if self.S_min <= t_cur <= self.S_max else 0
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * self.S_noise * torch.randn_like(x_cur)

            # Euler step.
            data.padded_sigmas = torch.tensor([t_hat]).reshape(-1,1).repeat(batch_size, 1).to(self.device)

            c_in = 1 / (self.sigma_data ** 2 + data.padded_sigmas ** 2).sqrt()
            data['ligand'].pos = c_in[batch] * copy.deepcopy(x_hat)
            data.noised_atom_pos = copy.deepcopy(x_hat)
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=dtype):
                denoised = self(data, dtype)

            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                data['ligand'].pos = c_in[batch] * copy.deepcopy(x_next)
                data.noised_atom_pos = copy.deepcopy(x_next)
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=dtype):
                    denoised = self(data, dtype)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            x_latents.append(x_next)

        return x_latents, x_next

    def forward(self, data, dtype=torch.float32):

        x, sigma = data.noised_atom_pos, data.padded_sigmas

        batch = data['ligand'].batch
        c_skip = self.sigma_data ** 2 / (sigma[batch] ** 2 + self.sigma_data ** 2)
        c_out = sigma[batch] * self.sigma_data / (sigma[batch] ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma[batch] ** 2).sqrt()
        c_noise = (sigma/self.sigma_data).log() / 4

        data['ligand'].pos = c_in * x

        F_x = self.net(data, c_noise, dtype)

        D_x = c_skip * x +  c_out * F_x

        return D_x

@register_model
def en_score_model_l1_4M_drop01(device, lm_embedding_type, **kwargs):
    model = EGNNModel(
        device=device, in_lig_edge_features=4,
        sigma_embed_dim=32, in_node_nf=128, in_dim=32,
        num_layers=6, lig_max_radius=5,
        rec_max_radius=30, cross_max_distance=250,
        distance_embed_dim=32,  cross_distance_embed_dim=32,
        batch_norm=True,
        dropout=0.1, lm_embedding_type=lm_embedding_type)

    return model


@register_model
def en_score_model_l1_21M_drop01(device, lm_embedding_type, **kwargs):
    model = EGNNModel(
        device=device, in_lig_edge_features=4,
        sigma_embed_dim=64, in_node_nf=256, in_dim=64,
        num_layers=7, lig_max_radius=5,
        rec_max_radius=30, cross_max_distance=250,
        distance_embed_dim=64,  cross_distance_embed_dim=64,
        batch_norm=True,
        dropout=0.1, lm_embedding_type=lm_embedding_type)

    return model