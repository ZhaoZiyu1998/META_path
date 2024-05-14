import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv
from . import BaseModel, register_model
from ..layers.macro_layer.SemanticConv import SemanticAttention
from ..layers.MetapathConv import MetapathConv
from ..utils.utils import extract_metapaths, get_ntypes_from_canonical_etypes
import pdb
@register_model('MLP')
class MLP(BaseModel):
    """
    This model replaces the HAN architecture with a 2-layer MLP for node feature transformation. It directly processes
    node features without utilizing the graph structure or meta-paths for aggregation.

    Parameters
    ------------
    in_dim : int
        Input feature dimension.
    hidden_dim : int
        Dimension of the hidden layer in MLP.
    out_dim : int
        Output feature dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(self, hidden_dim, out_dim, dropout,ntype_meta_paths_dict):
        pdb.set_trace()
        super(MLP, self).__init__()
        self.linear1 = nn.LazyLinear(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, h_dict):
        """
        Forward pass of MLP.

        Parameters
        -----------
        h_dict : dict[str, Tensor]
            The input features. Dict from node type to node features.

        Returns
        --------
        out_dict : dict[str, Tensor]
            The output features. Dict from node type to node features.
        """
        out_dict = {}
        for ntype, h in h_dict.items():
            pdb.set_trace()
            x = F.relu(self.linear1(h))
            x = self.dropout(x)
            x = self.linear2(x)
            out_dict[ntype] = x
        return out_dict

    @classmethod
    def build_model_from_args(cls, args, hg):
        """
        Class method to create an instance of MLP from command line arguments and the heterogeneous graph.

        Parameters
        ----------
        args : argparse.Namespace
            Command line arguments.
        hg : DGLHeteroGraph
            The heterogeneous graph.

        Returns
        -------
        An instance of MLP.
        """
        ntypes = set()
        if hasattr(args, 'target_link'):
            ntypes = get_ntypes_from_canonical_etypes(args.target_link)
        elif hasattr(args, 'category'):
            ntypes.add(args.category)
        else:
            raise ValueError

        ntype_meta_paths_dict = {}
        for ntype in ntypes:
            ntype_meta_paths_dict[ntype] = {}
            for meta_path_name, meta_path in args.meta_paths_dict.items():
                # a meta path starts with this node type
                if meta_path[0][0] == ntype:
                    ntype_meta_paths_dict[ntype][meta_path_name] = meta_path
        for ntype, meta_paths_dict in ntype_meta_paths_dict.items():
            if len(meta_paths_dict) == 0:
                ntype_meta_paths_dict[ntype] = extract_metapaths(ntype, hg.canonical_etypes)
        return cls(ntype_meta_paths_dict=ntype_meta_paths_dict,
            hidden_dim=args.hidden_dim,
                   out_dim=args.out_dim,
                   dropout=args.dropout)
