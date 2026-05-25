import torch
import torch.nn as nn
from torch.nn.modules.module import Module
import manifolds
from torch_scatter import scatter_mean
# from utils.helper import default_device
from manifolds import PoincareBall
from collections import defaultdict

class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, c_in, network, num_layers):
        super(HyperbolicGraphConvolution, self).__init__()
        self.agg = HypAgg(manifold, c_in, out_features, network, num_layers)

    def forward(self, input):
        x,adj= input
        h = self.agg.forward(x, adj)
        output = h, adj
        return output


class StackGCNs(Module):

    def __init__(self, num_layers, c, emb_size):
        super(StackGCNs, self).__init__()
        self.manifold = getattr(manifolds, "PoincareBall")()
        self.num_gcn_layers = num_layers - 1
        self.c = c
        self.emb_size = emb_size
        # self.gate1 = nn.Linear(self.emb_size, self.emb_size, bias=False).to(default_device())
        # self.gate2 = nn.Linear(self.emb_size, self.emb_size, bias=False).to(default_device())
        # self.sigmoid = nn.Sigmoid().to(default_device())
        self.order_attention = nn.Sequential(
            nn.Linear(4, 1),
            nn.ReLU(inplace=True),
            nn.Linear(1, 4),
            nn.Sigmoid()
        ).cuda()

    def plainGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            output.append(torch.spmm(adj, output[i]))
        return output[-1]

    def attentive_output(self, x):
        x_permute = x.permute(0, 2, 1).to(x.device)
        x_att_permute = self.order_attention(x_permute)
        att = x_att_permute.permute(0, 2, 1)
        x = att*x
        return x.sum(dim=1)

    def resAttSumGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            output.append(torch.spmm(adj, output[i]))
        output = torch.cat([res.unsqueeze(1) for res in output], dim=1)
        att_output = self.attentive_output(output)
        return att_output

    # def resSumGCN(self, inputs):
    #     x,entity_embedding, adj, edge_index, edge_type, relation_weight,num_users,num_items,n_entities = inputs
    #     output_inter = [self.manifold.logmap0(x,self.c)]
    #     output_graph = [self.manifold.logmap0(entity_embedding,self.c)]
    #     for i in range(self.num_gcn_layers):
    #         x, entity_embedding = self.aggregate(x,entity_embedding,adj, edge_index, edge_type, relation_weight,num_users,num_items,n_entities)
            
    #         output_inter.append(self.manifold.logmap0(x,self.c))
    #         output_graph.append(self.manifold.logmap0(entity_embedding,self.c))
            
            
    #     return sum(output_inter[1:]), sum(output_graph[1:])

    def resSumGCN(self, inputs):
        x_tangent, adj = inputs
        if self.num_gcn_layers == 0:
            return x_tangent
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            output.append(torch.spmm(adj, output[i]))
        return torch.clamp(sum(output[1:]), max=50)

    
    def resAddGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        if self.num_gcn_layers == 1:
            return torch.spmm(adj, x_tangent)
        for i in range(self.num_gcn_layers):
            if i == 0:
                output.append(torch.spmm(adj, output[i]))
            else:
                output.append(output[i] + torch.spmm(adj, output[i]))
        return output[-1]

    def denseGCN(self, inputs):
        x_tangent, adj = inputs
        output = [x_tangent]
        for i in range(self.num_gcn_layers):
            if i > 0:
                output.append(sum(output[1:i + 1]) + torch.spmm(adj, output[i]))
            else:
                output.append(torch.spmm(adj, output[i]))
        return output[-1]


class HypAgg(Module):
    """
    Hyperbolic aggregation layer.
    """

    def __init__(self, manifold, c, in_features, network, num_layers):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c
        self.in_features = in_features
        self.stackGCNs = getattr(StackGCNs(num_layers, c, self.in_features), network)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c)

        output= self.stackGCNs((x_tangent,adj))
        output = self.manifold.proj(self.manifold.expmap0(output, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)



class MultiHyperbolicGraphConvolution(nn.Module):
    """
    Multi Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold: PoincareBall, c_list, agg_mode, space2entities):
        super().__init__()
        self.aggs = nn.ModuleList([MultiHypFusionAgg(manifold, c, agg_mode, space2entities) for c in c_list])

    def forward(self, x: dict, adj: dict, entity2space_weights: dict):
        if len(self.aggs) == 0:
            return x
        h_list = [x]
        for i, agg in enumerate(self.aggs):
           h_list.append(agg(h_list[i], adj, entity2space_weights))
        # Add the outputs of each hgcn layer of the entity
        # e.g.: [layer1: [A1, B1], layer2: [A2, B2], ...] -> {A: A1 + A2 + ..., B: B1 + B2 + ...}
        output = dict((entity, sum([h[entity] for h in h_list[1:]])) for entity in x.keys())
        return output


class MultiHypFusionAgg(nn.Module):
    """
    Multiple Hyperbolic aggregation layer using space fusion.
    """
    def __init__(self, manifold: PoincareBall, c, agg_mode, space2entities):
        super().__init__()
        self.manifold = manifold
        self.c = c
        self.agg_fuc = dict((space, getattr(self, agg_mode.get(space, None)) if agg_mode.get(space, None) is not None else self.hyper_agg) \
                            for space in c.keys())
        self.space2entities = space2entities

    def forward(self, x: dict, adj: dict, entity2space_weights: dict):
        agg_embeddings_dict = defaultdict(int)
        for space in self.c.keys():
            e1, e2 = self.space2entities[space]
            embeddings_e1 = x[e1]
            x_input = embeddings_e1
            if e1 != e2:
                x_input = torch.cat([x_input, x[e2]], dim=0)
            x_agg = self.agg_fuc[space](x_input, self.c[space], adj[space])
            # Every aggregation result will mutiply a space weight
            agg_embeddings_dict[e1] += entity2space_weights[e1][space] * x_agg[:len(embeddings_e1), :]
            if e1 != e2:
                agg_embeddings_dict[e2] += entity2space_weights[e2][space] * x_agg[len(embeddings_e1):, :]

        return agg_embeddings_dict

    def hyper_agg(self, x, c, adj, pro=None):
        """
        Hyperbolic aggregation.
        First, map ``x`` to hyperbolic space by ``c``. Then aggregate ``x`` depending on ``adj``. 
        Lastly, map aggregation result back to euclidian space.
        """
        if pro is not None:
            x = torch.tanh(pro(x)) + x
        x_hyp = self.manifold.expmap0(x, c)
        x_hyp_agg = self.manifold.weighted_midpoint_spmm(x_hyp, c, adj)
        x_euc_agg = self.manifold.logmap0(x_hyp_agg, c)
        return x_euc_agg

    def eucli_agg(self, x, c, adj, pro=None):
        """
        Euclidian aggregation.
        Aggregate ``x`` depending on ``adj`` in euclidian space directly.
        """
        if pro is not None:
            x = torch.tanh(pro(x)) + x
        x_euc_agg = torch.sparse.mm(adj, x)
        return x_euc_agg

    def extra_repr(self):
        return 'c={}'.format(self.c)