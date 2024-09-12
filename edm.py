import e3nn
import math
import torch
import numpy as np
from e3nn import o3
from torch import nn
from .batchnorm import BatchNorm
from .registry import register_model
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter, scatter_mean

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
        self.num_categorical_features = feature_dims
        self.num_scalar_features = sigma_embed_dim
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


class TensorProductConvLayer(nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True, dropout=0.0,
                 hidden_features=None):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, tp.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean', dtype=torch.float32):

        edge_src, edge_dst = edge_index

        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out.to(torch.float32))
            out = out.to(dtype)
        return out

class TensorProductScoreModel(nn.Module):
    def __init__(self, device, in_lig_edge_features=4, sigma_embed_dim=32, sh_lmax=2,
                 ns=16, nv=4, num_conv_layers=2, lig_max_radius=5, rec_max_radius=30,
                 cross_max_distance=250, distance_embed_dim=32, cross_distance_embed_dim=32,
                 use_second_order_repr=False, batch_norm=True, dropout=0.0, lm_embedding_type=None):
        super(TensorProductScoreModel, self).__init__()
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.distance_embed_dim = distance_embed_dim
        self.cross_distance_embed_dim = cross_distance_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.ns, self.nv = ns, nv
        self.device = device
        self.time_embed = FourierEmbedding(self.sigma_embed_dim)

        self.num_conv_layers = num_conv_layers
        self.lig_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=8, sigma_embed_dim=sigma_embed_dim)
        self.lig_edge_embedding = nn.Sequential(nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim, ns),nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.rec_node_embedding = AtomEncoder(emb_dim=ns, feature_dims=8, sigma_embed_dim=sigma_embed_dim, lm_embedding_type=lm_embedding_type)
        self.rec_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))

        self.cross_edge_embedding = nn.Sequential(nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns), nn.ReLU(), nn.Dropout(dropout),nn.Linear(ns, ns))
        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)

        if use_second_order_repr:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o'
            ]

        self.lig_conv_layers, self.rec_conv_layers, self.lig_to_rec_conv_layers, self.rec_to_lig_conv_layers = \
        nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        for i in range(num_conv_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            parameters = {
                'in_irreps': in_irreps,
                'sh_irreps': self.sh_irreps,
                'out_irreps': out_irreps,
                'n_edge_features': 3 * ns,
                'hidden_features': 3 * ns,
                'residual': False,
                'batch_norm': batch_norm,
                'dropout': dropout
            }

            lig_layer = TensorProductConvLayer(**parameters)
            self.lig_conv_layers.append(lig_layer)

            rec_to_lig_layer = TensorProductConvLayer(**parameters)
            self.rec_to_lig_conv_layers.append(rec_to_lig_layer)

            if i != num_conv_layers - 1:
              rec_layer = TensorProductConvLayer(**parameters)
              self.rec_conv_layers.append(rec_layer)
              lig_to_rec_layer = TensorProductConvLayer(**parameters)
              self.lig_to_rec_conv_layers.append(lig_to_rec_layer)

        self.last_conv = TensorProductConvLayer(
            in_irreps=self.lig_conv_layers[-1].out_irreps,
            sh_irreps=self.sh_irreps,
            out_irreps=f'1x1o',
            n_edge_features=3 * ns,
            residual=False,
            dropout=dropout,
            batch_norm=batch_norm
        )

    def forward(self, data, times, dtype):

        # build ligand graph
        lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.build_lig_conv_graph(data, times)
        lig_src, lig_dst = lig_edge_index
        lig_node_attr = self.lig_node_embedding(lig_node_attr, dtype=dtype)
        lig_edge_attr = self.lig_edge_embedding(lig_edge_attr)

        # build receptor graph
        rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.build_rec_conv_graph(data, times)
        rec_src, rec_dst = rec_edge_index
        rec_node_attr = self.rec_node_embedding(rec_node_attr, dtype=dtype)
        rec_edge_attr = self.rec_edge_embedding(rec_edge_attr)

        # build cross graph
        cross_cutoff = data.sigma * 3 + 20
        cross_edge_index, cross_edge_attr, cross_edge_sh = self.build_cross_conv_graph(data, cross_cutoff)
        cross_lig, cross_rec = cross_edge_index
        cross_edge_attr = self.cross_edge_embedding(cross_edge_attr)

        for l in range(len(self.lig_conv_layers)):
            # intra graph message passing
            lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_src, :self.ns], lig_node_attr[lig_dst, :self.ns]], -1)

            lig_intra_update = self.lig_conv_layers[l](lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh, dtype=dtype)
            # inter graph message passing
            rec_to_lig_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)
            lig_inter_update = self.rec_to_lig_conv_layers[l](rec_node_attr, cross_edge_index, rec_to_lig_edge_attr_, cross_edge_sh,
                                                              out_nodes=lig_node_attr.shape[0], dtype=dtype)

            if l != len(self.lig_conv_layers) - 1:
                rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_src, :self.ns], rec_node_attr[rec_dst, :self.ns]], -1)
                rec_intra_update = self.rec_conv_layers[l](rec_node_attr, rec_edge_index, rec_edge_attr_, rec_edge_sh, dtype=dtype)

                lig_to_rec_edge_attr_ = torch.cat([cross_edge_attr, lig_node_attr[cross_lig, :self.ns], rec_node_attr[cross_rec, :self.ns]], -1)

                rec_inter_update = self.lig_to_rec_conv_layers[l](lig_node_attr, torch.flip(cross_edge_index, dims=[0]), lig_to_rec_edge_attr_,
                                                                  cross_edge_sh, out_nodes=rec_node_attr.shape[0], dtype=dtype)

            # padding original features
            lig_node_attr = F.pad(lig_node_attr, (0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1]))

            # update features with residual updates
            lig_node_attr = lig_node_attr + lig_intra_update + lig_inter_update

            if l != len(self.lig_conv_layers) - 1:
                rec_node_attr = F.pad(rec_node_attr, (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]))
                rec_node_attr = rec_node_attr + rec_intra_update + rec_inter_update

        lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_src, :self.ns], lig_node_attr[lig_dst, :self.ns]], -1)
        pos_pred = self.last_conv(lig_node_attr, lig_edge_index, lig_edge_attr_, lig_edge_sh, dtype=dtype)
        return pos_pred


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
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        node_attr = torch.cat([data['ligand'].pos, data['ligand'].x, data['ligand'].node_sigma_emb], 1)
        src, dst = edge_index
        edge_vec = data['ligand'].pos[dst.long()] - data['ligand'].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')
        return node_attr, edge_index, edge_attr, edge_sh

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
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh


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
        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return edge_index, edge_attr, edge_sh

class EquivariantElucidatedDiffusion(torch.nn.Module):
    def __init__(
        self,
        net: TensorProductScoreModel,
        sigma_min = 0.002,     # min noise level
        sigma_max = 80,        # max noise level
        sigma_data = 16,      # standard deviation of data distribution
    ):
        super().__init__()
        self.net = net
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        self.register_buffer('zero', torch.tensor(0.), persistent = False)

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
def e3_score_model_l1_4M_drop01(device, lm_embedding_type, **kwargs):
    model = TensorProductScoreModel(
        device=device, in_lig_edge_features=4,
        sigma_embed_dim=32, sh_lmax=2, ns=32,
        nv=8, num_conv_layers=5, lig_max_radius=5,
        rec_max_radius=30, cross_max_distance=250,
        distance_embed_dim=32,  cross_distance_embed_dim=32,
        use_second_order_repr=False,  batch_norm=True,
        dropout=0.1, lm_embedding_type=lm_embedding_type)

    return model

@register_model
def e3_score_model_l1_4M_drop00(device, lm_embedding_type, **kwargs):
    model = TensorProductScoreModel(
        device=device, in_lig_edge_features=4,
        sigma_embed_dim=32, sh_lmax=2, ns=32,
        nv=8, num_conv_layers=5, lig_max_radius=5,
        rec_max_radius=30, cross_max_distance=80,
        distance_embed_dim=32,  cross_distance_embed_dim=32,
        use_second_order_repr=False,  batch_norm=True,
        dropout=0.0, lm_embedding_type=lm_embedding_type)

    return model

