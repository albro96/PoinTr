import torch
from torch import nn

from pointnet2_ops import pointnet2_utils
from .SequenceTransformer import PCTransformer
from .build import MODELS

from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_farthest_points as fps

    
class Fold(nn.Module):
    def __init__(self, in_channel, step, hidden_dim=512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step

        a = (
            torch.linspace(-1.0, 1.0, steps=step, dtype=torch.float)
            .view(1, step)
            .expand(step, step)
            .reshape(1, -1)
        )
        b = (
            torch.linspace(-1.0, 1.0, steps=step, dtype=torch.float)
            .view(step, 1)
            .expand(step, step)
            .reshape(1, -1)
        )
        # self.folding_seed = torch.cat([a, b], dim=0).cuda()
        self.folding_seed = torch.cat([a, b], dim=0)

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, 3, 1),
        )

    def forward(self, x):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(
            bs, self.in_channel, num_sample
        )
        seed = (
            self.folding_seed.view(1, 2, num_sample)
            .expand(bs, 2, num_sample)
            .to(x.device)
        )

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2


@MODELS.register_module()
class TST(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.trans_dim
        self.knn_layer = config.knn_layer
        self.num_pred = config.num_pred
        self.num_query = config.num_query
        self.gt_type = config.get("gt_type", "full")
        self.config = config
        
        self.fold_step = int(pow(self.num_pred // self.num_query, 0.5) + 0.5)
        self.base_model = PCTransformer(
            in_chans=3,
            embed_dim=self.trans_dim,
            depth=[6, 8],
            drop_rate=0.0,
            num_query=self.num_query,
            knn_layer=self.knn_layer,
        )

        self.foldingnet = Fold(
            self.trans_dim, step=self.fold_step, hidden_dim=256
        )  # rebuild a cluster point

        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
        )
        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)

    def get_loss(self, ret, gt, config=None, epoch=0, loss_func=None, square=False):

        if loss_func is None:
            loss_coarse = chamfer_distance(ret[0], gt, norm=2, single_directional=True)[0]
            loss_fine = chamfer_distance(ret[1], gt, norm=2)[0]/2
            if not square:
                loss_coarse = torch.sqrt(loss_coarse)
                loss_fine = torch.sqrt(loss_fine)
        else:
            assert config is not None, "config is required for InfoCD"
            loss_coarse = loss_func(ret[0], gt, single_directional=True, config=config.InfoCD)
            loss_fine = loss_func(ret[1], gt, config=config.InfoCD)

        return loss_coarse, loss_fine

    def forward(self, allteeth, fdi, corrmask, gtmask, trafo, anta):
        """
        allteeth: [B, 28, 2, 2048, 3] (points in channel 0, normals in channel 1)
        fdi:      [B, 28]
        trafo:    [B, 28, 7]
        corrmask, gtmask, anta: other inputs for later use
        """
        B = allteeth.size(0)
        T = allteeth.size(1)  # fixed to 28 teeth
        
        # Select only the point coordinates (channel 0)
        # Resulting shape: [B, 28, 2048, 3]
        tooth_points = allteeth[:, :, 0, :, :]
        
        # Flatten batch and teeth so each tooth is processed independently:
        tooth_points = tooth_points.view(B * T, 2048, 3).transpose(1, 2)  # [B*28, 3, 2048]
        
        # Pass through a lightweight Conv1d-based extractor to get a feature vector per tooth.
        tooth_feats = self.tooth_extractor(tooth_points)  # [B*28, trans_dim, 2048]
        tooth_feats = torch.max(tooth_feats, dim=2)[0]      # [B*28, trans_dim]
        tooth_feats = tooth_feats.view(B, T, -1)            # [B, 28, trans_dim]
        
        # Get FDI embeddings and transform trafo (e.g., project 7D into D_dim)
        fdi_embed = self.fdi_embedding(fdi.long())          # [B, 28, fdi_dim]
        trafo_embed = self.trafo_proj(trafo)                  # [B, 28, trafo_dim]
        
        # Concatenate to build token for each tooth:
        tokens = torch.cat([tooth_feats, fdi_embed, trafo_embed], dim=-1)  # [B, 28, token_dim]
        
        # Feed the dental tokens to the transformer backbone.
        # Assume that PCTransformer (base_model) has been adjusted to accept token sequences.
        q, coarse_points = self.base_model(tokens)
        
        # (Rest of the network, such as the foldingNet to produce the dense output, remains similar.)
        B, M, C = q.shape
        global_feature = self.increase_dim(q.transpose(1, 2)).transpose(1, 2)
        global_feature = torch.max(global_feature, dim=1)[0]
        rebuild_feature = torch.cat(
            [global_feature.unsqueeze(-2).expand(-1, M, -1), q, coarse_points],
            dim=-1,
        )
        rebuild_feature = self.reduce_map(rebuild_feature.reshape(B * M, -1))
        relative_xyz = self.foldingnet(rebuild_feature).reshape(B, M, 3, -1)
        rebuild_points = (
            (relative_xyz + coarse_points.unsqueeze(-1))
            .transpose(2, 3)
            .reshape(B, -1, 3)
        )
        ret = (coarse_points, rebuild_points)
        return ret
