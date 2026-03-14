import torch

from torch import nn
from typing import List, Optional

from gmn.encoders import NodeEdgeFeatEncoder
from gmn.graph_models import EdgeMPNNDiT
from gmn.graph_pooling import GNNwEdgeReadout, DSEdgeReadout, DSNodeEdgeReadout


class GNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        num_gnn_layers: int = 4, 
        pre_encoder: bool = True, 
        undirected: bool = False, 
        readout_layers: int = 2, 
        pool_type: str = "ds", 
        pre_encoder_norms: bool = False, 
        pre_encoder_post_activation: bool = False, 
        pre_encoder_ff: bool = False, 
        readout_mode: Optional[str] = "pre",  # "pre" or "post"
        readout_hidden_dims: List[int] = []
    ):
        super().__init__()
        
        self.pre_encoder = NodeEdgeFeatEncoder(
            hidden_dim, 
            pre_encoder_norms, 
            pre_encoder_post_activation, 
            ff=pre_encoder_ff
        ) if pre_encoder else None
        
        self.undirected = undirected
        node_in_dim = hidden_dim if pre_encoder else 1
        edge_in_dim = hidden_dim if pre_encoder else 1

        gnn = EdgeMPNNDiT(
            node_in_dim, 
            edge_in_dim, 
            hidden_dim, 
            hidden_dim, 
            num_layers=num_gnn_layers, 
            dropout=0.0
        )
        readout_in_dim = hidden_dim

        if pool_type == "ds":
            readout = DSEdgeReadout(
                readout_in_dim, 
                out_dim, 
                readout_hidden_dims, 
                pre_pool="pre" in readout_mode
            )
        elif pool_type == "node_edge_ds":
            readout = DSNodeEdgeReadout(
                hidden_dim, 
                hidden_dim, 
                out_dim, 
                reduce="mean", 
                num_layers=readout_layers
            )
        else:
            raise ValueError("Invalid pooling type")

        self.gnn_w_readout = GNNwEdgeReadout(
            gnn, 
            readout, 
            use_nodes="node" in pool_type
        )

    def forward(self, x, edge_index, edge_attr, batch):
        if self.pre_encoder is not None:
            x, edge_attr = self.pre_encoder(x, edge_attr)
        else:
            x = torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
            edge_attr = edge_attr[:, 0].unsqueeze(1)

        if self.undirected:
            edge_index_t = edge_index.clone()
            edge_index_t[0], edge_index_t[1] = edge_index[1], edge_index[0]
            edge_index = torch.cat((edge_index.clone(), edge_index_t), 1)
            edge_attr = torch.cat((edge_attr.clone(), edge_attr.clone()), 0)

        graph_feat = self.gnn_w_readout(x, edge_index, edge_attr, batch)
        
        return graph_feat
