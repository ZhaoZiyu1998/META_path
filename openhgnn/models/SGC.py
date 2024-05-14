import torch
import torch.nn as nn
import dgl
from . import BaseModel, register_model
from ..utils.utils import extract_metapaths, get_ntypes_from_canonical_etypes
from dgl.nn.pytorch.conv import SGConv

@register_model('SGC')
class SGC(BaseModel):
    # def __init__(self, hg, in_dim, hidden_dim, out_dim, dropout, k):
    #     super(SGC, self).__init__()
    #     self.in_dim = in_dim
    #     self.hidden_dim = hidden_dim
    #     self.out_dim = out_dim
    #     self.k = 4
    #     self.dropout = nn.Dropout(dropout)

    #     # Initialize a ModuleDict to hold SGConv layers for each meta path of each node type
    #     self.sg_layers = nn.ModuleDict()

    #     # Extract and initialize SGConv layers for meta paths of each node type
    #     ntypes = hg.ntypes
    #     #import pdb;pdb.set_trace()
    #     self.ntype_meta_paths_dict = {}
    #     for ntype in ntypes:
    #         # Extract meta paths for the current node type
    #         meta_paths_dict = extract_metapaths(ntype, hg.canonical_etypes)
    #         self.ntype_meta_paths_dict[ntype] = meta_paths_dict
    #         #print(self.ntype_meta_paths_dict[ntype])
    #         for meta_path_name, meta_path in meta_paths_dict.items():
    #             # Initialize SGConv layer for each meta path
    #             layer_name = f"{ntype}_{meta_path_name}"
    #             self.sg_layers[layer_name] = SGConv(in_dim, hidden_dim, k=k, cached=True)
    #     #import pdb;pdb.set_trace()
    #     # Output layer
    #     self.output_layer = nn.Linear(hidden_dim, out_dim)
    # def forward(self, hg, h_dict):
    #     # Placeholder for aggregated features from different meta paths
    #     meta_path_features = {ntype: 0 for ntype in h_dict}
        
    #     for ntype, meta_paths in self.ntype_meta_paths_dict.items():
    #         for meta_path_name, meta_path in meta_paths.items():
    #             # Generate meta path specific subgraph
    #             print(meta_path)
    #             subgraph = dgl.metapath_reachable_graph(hg, meta_path)
    #             subgraph = dgl.add_self_loop(subgraph)
    #             # Get node features for the current meta path
    #             features = h_dict[ntype]

    #             # Apply SGC for the current meta path
    #             sgc_layer_key = f"{ntype}_{meta_path_name}"
    #             if sgc_layer_key in self.sg_layers:
    #                 features_transformed = self.sg_layers[sgc_layer_key](subgraph, features)
    #                 meta_path_features[ntype] += features_transformed
    #     import pdb;pdb.set_trace
    #     # Apply dropout and output layer to aggregated features
    #     for ntype in meta_path_features:
    #         meta_path_features[ntype] = self.dropout(meta_path_features[ntype])
    #         meta_path_features[ntype] = self.output_layer(meta_path_features[ntype])

    #     return meta_path_features

    def __init__(self, hg, in_dim, hidden_dim, out_dim, dropout, k, selected_metapath=None):
        super(SGC, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.k = 4
        self.dropout = nn.Dropout(dropout)
        
        # Selected metapath for all node types
        self.selected_metapath = selected_metapath

        # Initialize SGConv layer for the selected metapath
        self.sg_conv = SGConv(in_dim, hidden_dim, k=k, cached=True)

        # Output layer
        self.output_layer = nn.ModuleDict({
            ntype: nn.Linear(hidden_dim, out_dim) for ntype in hg.ntypes
        })

    def forward(self, hg, h_dict):
        
        # Generate meta path specific subgraph
        if self.selected_metapath is not None:
            import pdb;pdb.set_trace()
            subgraph = dgl.metapath_reachable_graph(hg, self.selected_metapath)
            subgraph = dgl.add_self_loop(subgraph)
        else:
            raise ValueError("Selected metapath is not defined.")

        meta_path_features = {}
        for ntype, features in h_dict.items():
            # Apply SGConv for the nodes in the subgraph
            if ntype in subgraph.ntypes:
                features_transformed = self.sg_conv(subgraph, features)
                features_transformed = self.dropout(features_transformed)
                meta_path_features[ntype] = self.output_layer[ntype](features_transformed)


        return meta_path_features

    @classmethod
    def build_model_from_args(cls, args, hg):
        # Specify the selected metapath
        #{'author': {'mp0': [('author', 'author-paper', 'paper'), ('paper', 'paper-author', 'author')]}, 'paper': {'mp0': [('paper', 'paper-author', 'author'), ('author', 'author-paper', 'paper')], 'mp1': [('paper', 'paper-subject', 'subject'), ('subject', 'subject-paper', 'paper')]}, 'subject': {'mp0': [('subject', 'subject-paper', 'paper'), ('paper', 'paper-subject', 'subject')]}}
       # {'mp0': [('subject', 'subject-paper', 'paper'), ('paper', 'paper-subject', 'subject')]}
        selected_metapath =  [('paper', 'paper-author', 'author'), ('author', 'author-paper', 'paper')]
        return cls(hg=hg, 
                   in_dim=args.in_dim, 
                   hidden_dim=args.hidden_dim, 
                   out_dim=args.out_dim, 
                   dropout=args.dropout, 
                   k=args.k,
                   selected_metapath=selected_metapath)
        