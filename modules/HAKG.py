import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.utils import softmax as scatter_softmax
from modules.model_utils import *
from manifolds import PoincareBall
import time
import math
import os
from datetime import datetime
import umap
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.sparse as sp
import manifolds
import layers.hyp_layers as hyp_layers
from utils.init import xavier_uniform_initialization
import re

def _sparse_dropout(x, rate=0.5):
    noise_shape = x._nnz()

    random_tensor = rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).type(torch.bool)
    i = x._indices()
    v = x._values()

    i = i[:, dropout_mask]
    v = v[dropout_mask]

    # out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
    # return out * (1. / (1 - rate))
        # 创建新的稀疏张量
    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    # 缩放非零值
    scaled_out = out * (1. / (1 - rate))

    # 获取最终的索引和非零值
    final_i = scaled_out._indices()
    final_v = scaled_out._values()

    return scaled_out, final_i, final_v

def _mae_edge_mask_adapt_mixed(edge_index, edge_type, topk_egde_id):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    topk_egde_id = topk_egde_id.cpu().numpy()
    topk_mask = np.zeros(n_edges, dtype=bool)
    topk_mask[topk_egde_id] = True
    # add another group of random mask
    random_indices = np.random.choice(
        n_edges, size=topk_egde_id.shape[0], replace=False)
    random_mask = np.zeros(n_edges, dtype=bool)
    random_mask[random_indices] = True
    # combine two masks
    mask = topk_mask | random_mask

    remain_edge_index = edge_index[:, ~mask]
    remain_edge_type = edge_type[~mask]
    masked_edge_index = edge_index[:, mask]
    masked_edge_type = edge_type[mask]

    return remain_edge_index, remain_edge_type, masked_edge_index, masked_edge_type, mask

def _edge_sampling(edge_index, edge_type, samp_rate=0.5):
    # edge_index: [2, -1]
    # edge_type: [-1]
    n_edges = edge_index.shape[1]
    random_indices = np.random.choice(
        n_edges, size=int(n_edges * samp_rate), replace=False)
    return edge_index[:, random_indices], edge_type[random_indices]

def _relation_aware_edge_sampling(edge_index, edge_type, n_relations, samp_rate=0.5):
    # exclude interaction
    for i in range(n_relations - 1):
        edge_index_i, edge_type_i = _edge_sampling(
            edge_index[:, edge_type == i], edge_type[edge_type == i], samp_rate)
        if i == 0:
            edge_index_sampled = edge_index_i
            edge_type_sampled = edge_type_i
        else:
            edge_index_sampled = torch.cat(
                [edge_index_sampled, edge_index_i], dim=1)
            edge_type_sampled = torch.cat(
                [edge_type_sampled, edge_type_i], dim=0)
    return edge_index_sampled, edge_type_sampled

class Aggregator(nn.Module):
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self, n_users, n_items,n_heads,d_k,c, manifold):
        super(Aggregator, self).__init__()
        self.n_users = n_users
        self.n_items = n_items

        self.gate1 = nn.Linear(64, 64, bias=False)
        self.gate2 = nn.Linear(64, 64, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.c = c
        self.manifold = manifold

        self.n_heads = n_heads
        self.d_k = d_k
    def KG_forward(self, entity_look_up_emb, edge_index, edge_type,
                   relation_lookup_emb,W_Q):
        head, tail = edge_index
        # exclude interact in IDs, remap [1, n_relations) to [0, n_relations-1)
        relation_type = edge_type - 1
        # [n_entities_, emb_dim]
        n_entities = entity_look_up_emb.shape[0]
        n_tripets = len(head)

        head_emb = entity_look_up_emb[head]
        tail_emb = entity_look_up_emb[tail]
        relation_emb = relation_lookup_emb[relation_type]

        # Mobius addition
        # map to hyperbolic point
        hyper_head_emb = self.manifold.expmap0(head_emb, self.c)

        # map to local hyperbolic point
        hyper_tail_emb = self.manifold.expmap(tail_emb, hyper_head_emb, self.c)
        hyper_relation_emb = self.manifold.expmap(relation_emb, hyper_head_emb, self.c)
        
        
        query = self.manifold.mobius_matvec(W_Q, hyper_head_emb, self.c).view(-1, self.n_heads, self.d_k)
        key = self.manifold.mobius_matvec(W_Q, hyper_tail_emb, self.c).view(-1, self.n_heads, self.d_k)
        
        # query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        # key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)

        if edge_type is not None:
            # key = key * self.relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)
            # key = key + self.relation_weight[edge_type - 1].view(-1, self.n_heads, self.d_k)
            key = self.manifold.proj(self.manifold.mobius_add(key, hyper_relation_emb.view(-1, self.n_heads, self.d_k), self.c), self.c).view(-1, self.n_heads, self.d_k)
        # edge_attn = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        edge_attn_score = -self.manifold.sqdist(query, key, self.c) / math.sqrt(self.d_k)
        
        neigh_relation_emb = self.manifold.proj(self.manifold.mobius_add(hyper_tail_emb, hyper_relation_emb, self.c), self.c)
        value = neigh_relation_emb.view(-1, self.n_heads, self.d_k)

        edge_attn_score = scatter_softmax(edge_attn_score, head)
        entity_agg = value * edge_attn_score.view(-1, self.n_heads, 1)
        res = entity_agg.view(-1, self.n_heads*self.d_k)
        
        
        # Mobius addition in hyperbolic space
        # res = project(mobius_add(hyper_tail_emb, hyper_relation_emb))
        # map to local tangent point
        res = self.manifold.logmap(hyper_head_emb, res, self.c)
        # res = tail_emb + relation_emb
        entity_agg = scatter_mean(src=res, index=head, dim_size=n_entities, dim=0)

        return entity_agg


    def KG_forward2(self, entity_look_up_emb, edge_index, edge_type,
                   relation_lookup_emb,W_Q):
        head, tail = edge_index
        # exclude interact in IDs, remap [1, n_relations) to [0, n_relations-1)
        relation_type = edge_type - 1
        # [n_entities_, emb_dim]
        n_entities = entity_look_up_emb.shape[0]
        n_tripets = len(head)

        head_emb = entity_look_up_emb[head]
        tail_emb = entity_look_up_emb[tail]
        relation_emb = relation_lookup_emb[relation_type]

        hyper_head_emb = self.manifold.expmap0(head_emb, self.c)

        # map to local hyperbolic point
        hyper_tail_emb = self.manifold.expmap(tail_emb, hyper_head_emb, self.c)
        hyper_relation_emb = self.manifold.expmap(relation_emb, hyper_head_emb, self.c)
        # Mobius addition in hyperbolic space
        res = self.manifold.proj(self.manifold.mobius_add(hyper_tail_emb, hyper_relation_emb, self.c), self.c)
        # map to local tangent point
        res = self.manifold.logmap(hyper_head_emb, res, self.c)
        # res = tail_emb + relation_emb
        entity_agg = scatter_mean(src=res, index=head, dim_size=n_entities, dim=0)

        return entity_agg
    
    def KG_forward_tan(self, entity_look_up_emb, edge_index, edge_type,
                   relation_lookup_emb,W_Q):
        head, tail = edge_index
        # exclude interact in IDs, remap [1, n_relations) to [0, n_relations-1)
        relation_type = edge_type - 1
        # [n_entities_, emb_dim]
        n_entities = entity_look_up_emb.shape[0]
        n_tripets = len(head)

        head_emb = entity_look_up_emb[head]
        tail_emb = entity_look_up_emb[tail]
        relation_emb = relation_lookup_emb[relation_type]

        res = tail_emb + relation_emb
        entity_agg = scatter_mean(src=res, index=head, dim_size=n_entities, dim=0)

        return entity_agg
    def forward(self, entity_emb, user_emb, item_emb_cf,
                edge_index, edge_type, interact_mat,
                relation_weight, W_Q, flag,hyper):

        """KG aggregate"""
        if hyper:
            if flag:
                entity_agg = self.KG_forward(entity_emb, edge_index, edge_type, relation_weight,W_Q)
            else:
                entity_agg = self.KG_forward2(entity_emb, edge_index, edge_type, relation_weight,W_Q)
        else:
            entity_agg = self.KG_forward_tan(entity_emb, edge_index, edge_type, relation_weight,W_Q)
        """user aggregate"""

        item_user_mat = interact_mat.transpose(0,1)
        item_agg_cf = torch.sparse.mm(item_user_mat, user_emb)
        
        user_item_mat = interact_mat
        
        item_emb_kg = entity_emb[:self.n_items]

        gi = self.sigmoid(self.gate1(item_emb_cf) + self.gate2(item_emb_kg))
        item_emb_fusion = (gi * item_emb_cf) + ((1 - gi) * item_emb_kg)
        user_agg = torch.sparse.mm(user_item_mat, item_emb_fusion)

        return entity_agg, user_agg, item_agg_cf, item_emb_fusion


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users,
                 n_items, n_relations, interact_mat,
                 device, c=None, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_items = n_items
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.device = device
        self.n_hops = n_hops
        
        # 支持可变曲率 c
        if c is None:
            self.c = torch.tensor([1.0]).to(self.device)
        else:
            self.c = c if isinstance(c, torch.Tensor) else torch.tensor([c]).to(self.device)
        
        # 初始化 PoincareBall manifold
        self.manifold = PoincareBall()
        
        self.W_Q = nn.Parameter(torch.Tensor(channel, channel))
        
        self.n_heads = 2
        self.d_k = channel // self.n_heads

        nn.init.xavier_uniform_(self.W_Q)

        relation_weight = nn.init.xavier_uniform_(torch.empty(n_relations - 1, channel))  # not include interact
        self.relation_weight = nn.Parameter(relation_weight)  # [n_relations - 1, in_channel]

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users, n_items=n_items, 
                                         n_heads=self.n_heads, d_k=self.d_k, c=self.c, 
                                         manifold=self.manifold).to(self.device))

        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))


    def forward(self, user_emb, entity_emb, item_emb_cf, edge_index, edge_type,
                interact_mat,flag,hyper, mess_dropout=True, node_dropout=False):

        """node dropout"""
        # if node_dropout:
        #     edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
        #     interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        item_emb_cf_res = item_emb_cf
        item_emb_fusion_res = item_emb_cf + entity_emb[:self.n_items]
        

        for i in range(len(self.convs)):
            entity_emb, user_emb, item_emb_cf,item_emb_fusion= self.convs[i](entity_emb, user_emb, item_emb_cf,
                                                 edge_index, edge_type, interact_mat,
                                                 self.relation_weight,self.W_Q,flag,hyper)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
                item_emb_cf = self.dropout(item_emb_cf)
                item_emb_fusion = self.dropout(item_emb_fusion)
            entity_emb = F.normalize(entity_emb)
            user_emb = F.normalize(user_emb)
            item_emb_cf = F.normalize(item_emb_cf)
            item_emb_fusion = F.normalize(item_emb_fusion)


            """result emb"""
            entity_res_emb = torch.add(entity_res_emb, entity_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)
            item_emb_cf_res = torch.add(item_emb_cf_res, item_emb_cf)
            item_emb_fusion_res = torch.add(item_emb_fusion_res, item_emb_fusion)


        return entity_res_emb, user_res_emb, item_emb_cf_res, item_emb_fusion_res
    
    @torch.no_grad()
    def norm_attn_computer(self, entity_emb, edge_index, edge_type=None,return_logits=False):
        head, tail = edge_index
        
        
        hyper_head_emb = self.manifold.expmap0(entity_emb[head], self.c)

        # map to local hyperbolic point
        hyper_tail_emb = self.manifold.expmap(entity_emb[tail], hyper_head_emb, self.c)
        hyper_relation_emb = self.manifold.expmap(self.relation_weight[edge_type - 1], hyper_head_emb, self.c)

        
        query = self.manifold.mobius_matvec(self.W_Q, hyper_head_emb, self.c).view(-1, self.n_heads, self.d_k)
        key = self.manifold.mobius_matvec(self.W_Q, hyper_tail_emb, self.c).view(-1, self.n_heads, self.d_k)
        
        # query = (entity_emb[head] @ self.W_Q).view(-1, self.n_heads, self.d_k)
        # key = (entity_emb[tail] @ self.W_Q).view(-1, self.n_heads, self.d_k)

        if edge_type is not None:
            # key = key * self.relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)
            # key = key + self.relation_weight[edge_type - 1].view(-1, self.n_heads, self.d_k)
            key = self.manifold.proj(self.manifold.mobius_add(key, hyper_relation_emb.view(-1, self.n_heads, self.d_k), self.c), self.c).view(-1, self.n_heads, self.d_k)
        # edge_attn = (query * key).sum(dim=-1) / math.sqrt(self.d_k)
        edge_attn = -self.manifold.sqdist(query, key, self.c) / math.sqrt(self.d_k)
        
        edge_attn_logits = edge_attn.mean(-1).detach()
        

        # softmax by head_node
        edge_attn_score = scatter_softmax(edge_attn_logits, head)
        # normalization by head_node degree
        norm = scatter_sum(torch.ones_like(head), head, dim=0, dim_size=entity_emb.shape[0])
        norm = torch.index_select(norm, 0, head)
        edge_attn_score = edge_attn_score * norm
        # print attn score
        # print("edge_attn_score std: ",edge_attn_score.std())
        if return_logits:
            return edge_attn_score, edge_attn_logits
        return edge_attn_score


class Recommender(nn.Module):
    def __init__(self, data_config, args_config, graph, adj_mat,adj_train,train_item_set):
        super(Recommender, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_relations = data_config['n_relations']
        self.n_entities = data_config['n_entities']  # include items
        self.n_nodes = data_config['n_nodes']  # n_users + n_entities

        self.margin_ccl = args_config.margin
        self.num_neg_sample = args_config.num_neg_sample

        self.decay = args_config.l2
        self.angle_loss_w = args_config.angle_loss_w
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.node_dropout = args_config.node_dropout
        self.node_dropout_rate = args_config.node_dropout_rate
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.loss_f = args_config.loss_f
        self.device = torch.device("cuda:" + str(args_config.gpu_id)) if args_config.cuda \
                                                                      else torch.device("cpu")
        
        
        self.lambda_1 = args_config.lambda1
        self.temp = args_config.temp
        self.dataset = args_config.dataset
        
                                                                      
        self.dropout_p = 0.5
        self.anlge_emb_dropout = nn.Dropout(p=self.dropout_p)
        # self.adj_mat = adj_mat
        self.adj_train = adj_train
        
        self.graph = graph
        
        self.edge_index, self.edge_type = self._get_edges(graph)
        self.triplet_item_att = self._triplet_sampling(self.edge_index, self.edge_type).t()


        dataset = args_config.dataset
        emb_user_path = os.path.join('save/poincare_cosine_sim', f'{dataset}_user_{args_config.emb}.pt')
        emb_item_path = os.path.join('save/poincare_cosine_sim', f'{dataset}_item_{args_config.emb}.pt')
        self.user_emb= nn.Embedding.from_pretrained(torch.load(emb_user_path), freeze=False)
        self.item_emb =  nn.Embedding.from_pretrained(torch.load(emb_item_path), freeze=False)
        
        initializer = nn.init.xavier_uniform_
        self.orther_item_emb = nn.Parameter(initializer(torch.empty(self.n_nodes - self.n_users - self.n_items, self.emb_size)))

        # self.all_embed = nn.Parameter(initializer(torch.empty(self.n_nodes, self.emb_size)))
        # self.item_emb_cf = nn.Parameter(initializer(torch.empty(self.n_items, self.emb_size)))
        self.item_emb_cf = nn.Embedding.from_pretrained(torch.load(emb_item_path), freeze=False)

        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.adj_train).to(self.device)

        
        self._init_loss_function()
        # self._init_entities_remap()

        self.mae_msize = getattr(args_config, 'mae_msize', 256)
        self.mae_coef = args_config.mae_loss_w
        self.cl_drop = 0.5
        self.samp_func = "torch"
        self.cl_coef = 0.1
        self.cl_coef2 = 0.01
        self.tau = 0.7
        self.train_item_set = train_item_set
        
        # 初始化 PoincareBall manifold 和曲率 c
        self.manifold = PoincareBall()
        self.c = torch.tensor([1.0]).to(self.device)
        self.gcn = self._init_model()


    # def _init_weight(self):
        

    
    
    def _init_model(self):
        return GraphConv(channel=self.emb_size,
                         n_hops=self.context_hops,
                         n_users=self.n_users,
                         n_relations=self.n_relations,
                         n_items=self.n_items,
                         interact_mat=self.interact_mat,
                         device=self.device,
                         c=self.c,
                         node_dropout_rate=self.node_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate,
                            )

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _get_edges(self, graph):
        graph_tensor = torch.tensor(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return index.t().long().to(self.device), type.long().to(self.device)

    def _init_loss_function(self):
        if self.loss_f == "inner_bpr":
            self.loss = self.create_inner_bpr_loss
        elif self.loss_f == "dis_bpr":
            self.loss = self.create_dis_bpr_loss
        elif self.loss_f == 'contrastive_loss':
            self.loss = self.create_contrastive_loss
        else:
            raise NotImplementedError

    def _triplet_sampling(self, edge_index, edge_type, rate=0.5):
        """
        edge_index: [2, E]
        edge_type:  [E]
        return:     [N, 2]  (h, t) pairs for angle loss
        """
        edge_index_t = edge_index.t()  # [E, 2]

        if self.dataset == 'mooccube_rel8':
            # mooccube_rel8: 选 head/tail 跨越 n_items 边界的边（user-item 或 item-entity）
            sample = []
            for idx, h_t in enumerate(edge_index_t):
                if (h_t[0] >= self.n_items and h_t[1] < self.n_items) or (h_t[0] < self.n_items and h_t[1] >= self.n_items):
                    sample.append(idx)
            sample = torch.LongTensor(sample).to(edge_index.device)
            return edge_index_t[sample]
        else:
            # 其他数据集（如 mooper）：按 hierarchy relation ids 筛选
            hier_rel_ids = {1, 4, 6, 7, 13}
            # 1  : course → chapter
            # 4  : discipline → course
            # 6  : discipline → exercise
            # 7  : exercise → challenge
            # 13 : course → exercise (inverse of exercise → course)
            sample_idx = []
            for idx in range(edge_type.size(0)):
                if edge_type[idx].item() in hier_rel_ids:
                    sample_idx.append(idx)
            sample_idx = torch.LongTensor(sample_idx).to(edge_index.device)
            return edge_index_t[sample_idx]

    def half_aperture(self, u):
        eps = 1e-6
        K = 0.1
        sqnu = u.pow(2).sum(dim=-1)
        sqnu.clamp_(min=0, max=1 - eps)
        return torch.asin((K * (1 - sqnu) / torch.sqrt(sqnu)).clamp(min=-1 + eps, max=1 - eps))

    def angle_at_u(self, u, v):
        eps = 1e-6
        norm_u = u.norm(2, dim=-1)
        norm_v = v.norm(2, dim=-1)
        dot_prod = (u * v).sum(dim=-1)
        edist = (u - v).norm(2, dim=-1)  # euclidean distance
        num = (dot_prod * (1 + norm_u ** 2) - norm_u ** 2 * (1 + norm_v ** 2))
        denom = (norm_u * edist * ((1 + norm_v ** 2 * norm_u ** 2 - 2 * dot_prod).clamp(min=eps).sqrt())) + eps
        return (num / denom).clamp_(min=-1 + eps, max=1 - eps).acos()

    def angle_loss(self, entity_emb, user):
        hier_hs = entity_emb[self.triplet_item_att[0]]
        hier_ts = entity_emb[self.triplet_item_att[1]]

        emb_drop = self.anlge_emb_dropout(torch.ones(size=hier_hs.shape)) * self.dropout_p  # need to tune
        emb_drop = emb_drop.to(self.device)
        hier_hs = hier_hs * emb_drop
        hier_ts = hier_ts * emb_drop

        loss3 = 0
        batch_size = user.shape[0]
        num = self.triplet_item_att.shape[1]
        num_x = math.ceil(num / batch_size)
        for i in range(num_x):
            hier_h = hier_hs[i * batch_size:(i + 1) * batch_size]
            hier_t = hier_ts[i * batch_size:(i + 1) * batch_size]
            angle_half = self.angle_at_u(hier_h, hier_t) - self.half_aperture(hier_h)
            angle_half[angle_half < 0] = 0
            loss3 += torch.sum(angle_half)

        loss3 = self.angle_loss_w * loss3 / num

        return loss3


    def forward(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items'].view(-1)

        user_emb = self.user_emb.weight
        entity_emb = torch.cat([self.item_emb.weight, self.orther_item_emb], dim=0)
        # user_emb = self.all_embed[:self.n_users, :]
        # entity_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        """node dropout"""
        # 1. graph sprasification;
        edge_index, edge_type = _relation_aware_edge_sampling(
            self.edge_index, self.edge_type, self.n_relations, self.node_dropout_rate)
        # 2. compute rationale scores;
        edge_attn_score, edge_attn_logits = self.gcn.norm_attn_computer(
            entity_emb, edge_index, edge_type,  return_logits=True)
        # for adaptive UI MAE
        item_attn_mean_1 = scatter_mean(edge_attn_score, edge_index[0], dim=0, dim_size=self.n_entities)
        item_attn_mean_1[item_attn_mean_1 == 0.] = 1.
        item_attn_mean_2 = scatter_mean(edge_attn_score, edge_index[1], dim=0, dim_size=self.n_entities)
        item_attn_mean_2[item_attn_mean_2 == 0.] = 1.
        item_attn_mean = (0.5 * item_attn_mean_1 + 0.5 * item_attn_mean_2)[:self.n_items]
        # for adaptive MAE training
        std = torch.std(edge_attn_score).detach()
        noise = -torch.log(-torch.log(torch.rand_like(edge_attn_score)))
        edge_attn_score = edge_attn_score + noise
        topk_v, topk_attn_edge_id = torch.topk(
            edge_attn_score, self.mae_msize, sorted=False)
        top_attn_edge_type = edge_type[topk_attn_edge_id]

        enc_edge_index, enc_edge_type, masked_edge_index, masked_edge_type, mask_bool = _mae_edge_mask_adapt_mixed(edge_index, edge_type, topk_attn_edge_id)

        inter_mat, inter_edge, inter_edge_w = _sparse_dropout(
            self.interact_mat, self.node_dropout_rate)
        
        entity_gcn_emb, user_gcn_emb, item_gcn_emb_cf,item_emb_fusion_res = self.gcn(user_emb,
                                                     entity_emb,
                                                     self.item_emb_cf.weight,
                                                     enc_edge_index,
                                                     enc_edge_type,
                                                     inter_mat,flag = True,hyper = True,
                                                     mess_dropout=self.mess_dropout,
                                                     node_dropout=self.node_dropout)
        
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        pos_e_cf, neg_e_cf = item_gcn_emb_cf[pos_item], item_gcn_emb_cf[neg_item]
        loss1 = self.loss(u_e, pos_e, neg_e, pos_e_cf, neg_e_cf)
        loss2 = self.angle_loss(entity_emb, user)
        # loss2 = 0
        
        # MAE task with dot-product decoder
        # mask_size, 2, channel
        node_pair_emb = entity_gcn_emb[masked_edge_index.t()]
        # mask_size, channel
        masked_edge_emb = self.gcn.relation_weight[masked_edge_type-1]
        mae_loss = self.mae_coef * \
            self.create_mae_loss(node_pair_emb, masked_edge_emb)
        # mae_loss = 0
        loss = loss1 + loss2 + mae_loss

        return loss

    def create_mae_loss(self, node_pair_emb, masked_edge_emb=None):
        head_embs, tail_embs = node_pair_emb[:, 0, :], node_pair_emb[:, 1, :]
        
        hyper_head_emb = self.manifold.expmap0(head_embs, self.c)

        # map to local hyperbolic point
        hyper_tail_emb = self.manifold.expmap(tail_embs, hyper_head_emb, self.c)
        hyper_relation_emb = self.manifold.expmap(masked_edge_emb, hyper_head_emb, self.c)

        if masked_edge_emb is not None:
            pos1 = self.manifold.proj(self.manifold.mobius_add(hyper_tail_emb, hyper_relation_emb, self.c), self.c)
            # pos1 = tail_embs * masked_edge_emb
        else:
            pos1 = tail_embs
            
        dist = self.manifold.sqdist(hyper_head_emb, pos1, self.c)
        # scores = (pos1 - head_embs).sum(dim=1).abs().mean(dim=0)
        # scores = - \
        #     torch.log(torch.sigmoid(torch.mul(pos1, head_embs).sum(1))).mean()
        scores = - \
            torch.log(torch.sigmoid(-dist)).mean()
        return scores
    
    def generate(self):
        user_emb = self.user_emb.weight
        entity_emb = torch.cat([self.item_emb.weight, self.orther_item_emb], dim=0)
        # user_emb = self.all_embed[:self.n_users, :]
        # entity_emb = self.all_embed[self.n_users:, :]
        entity_gcn_emb, user_gcn_emb, item_gcn_emb_cf,item_emb_fusion_res= self.gcn(user_emb,
                                                    entity_emb,
                                                    self.item_emb_cf.weight,
                                                    self.edge_index,
                                                    self.edge_type,
                                                    self.interact_mat,flag = True,hyper = True,
                                                    mess_dropout=False, node_dropout=False)
        # print('generate : ', np.isnan(entity_gcn_emb.detach().cpu().numpy()).any())
        entity_gcn_emb[:self.n_items] += item_gcn_emb_cf
        # self.plot_entity_embeddings(entity_gcn_emb)

        return entity_gcn_emb, user_gcn_emb

    def rating(self, u_g_embeddings, i_g_embeddings):
        if self.loss_f == "inner_bpr":
            return torch.matmul(u_g_embeddings, i_g_embeddings.t()).detach().cpu()

        elif self.loss_f == 'contrastive_loss':
            # u_g_embeddings = F.normalize(u_g_embeddings)
            # i_g_embeddings = F.normalize(i_g_embeddings)
            return torch.cosine_similarity(u_g_embeddings.unsqueeze(1), i_g_embeddings.unsqueeze(0), dim=2).detach().cpu()

        # else:
        #     n_user = len(u_g_embeddings)
        #     n_item = len(i_g_embeddings)
        #     hyper_rate_matrix = np.zeros(shape=(n_user, n_item))

        #     hyper_u_g_embeddings = expmap0(u_g_embeddings)
        #     hyper_i_g_embeddings = expmap0(i_g_embeddings)

        #     for i in range(n_user):
        #         # [1, dim]
        #         one_hyper_u = hyper_u_g_embeddings[i, :]
        #         # [n_item, dim]
        #         one_hyper_u = one_hyper_u.expand(n_item, -1)
        #         one_hyper_score = -1 * sq_hyp_distance(one_hyper_u, hyper_i_g_embeddings)
        #         hyper_rate_matrix[i, :] = one_hyper_score.squeeze().detach().cpu()

        #     return hyper_rate_matrix

    def create_contrastive_loss(self, u_e, pos_e, neg_e, pos_e_cf, neg_e_cf):
        batch_size = u_e.shape[0]

        u_e = F.normalize(u_e)
        pos_e = F.normalize(pos_e)
        neg_e = F.normalize(neg_e)
        pos_e_cf = F.normalize(pos_e_cf)
        neg_e_cf = F.normalize(neg_e_cf)

        ui_pos = torch.relu(2 - (torch.cosine_similarity(u_e, pos_e, dim=1) + torch.cosine_similarity(u_e, pos_e_cf, dim=1)))
        users_batch = torch.repeat_interleave(u_e, self.num_neg_sample, dim=0)

        ui_neg1 = torch.relu(torch.cosine_similarity(users_batch, neg_e, dim=1) - self.margin_ccl)
        ui_neg1 = ui_neg1.view(batch_size, -1)
        x = ui_neg1>0
        ui_neg_loss1 = torch.sum(ui_neg1,dim=-1)/(torch.sum(x, dim=-1) + 1e-5)

        ui_neg2 = torch.relu(torch.cosine_similarity(users_batch, neg_e_cf, dim=1) - self.margin_ccl)
        ui_neg2 = ui_neg2.view(batch_size, -1)
        x = ui_neg2 > 0
        ui_neg_loss2 = torch.sum(ui_neg2, dim=-1) / (torch.sum(x, dim=-1) + 1e-5)

        loss = ui_pos + ui_neg_loss1 + ui_neg_loss2

        return loss.mean()


    def create_inner_bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        cf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))
        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size

        return cf_loss + emb_loss

    # def create_dis_bpr_loss(self, users, pos_items, neg_items):
    #     hyper_users = expmap0(users)
    #     hyper_pos_items = expmap0(pos_items)
    #     hyper_neg_items = expmap0(neg_items)

    #     hyper_pos_dis = sq_hyp_distance(hyper_users, hyper_pos_items)
    #     hyper_neg_dis = sq_hyp_distance(hyper_users, hyper_neg_items)
    
    def create_dis_bpr_loss(self, users, pos_items, neg_items):
        hyper_users = self.manifold.expmap0(users, self.c)
        hyper_pos_items = self.manifold.expmap0(pos_items, self.c)
        hyper_neg_items = self.manifold.expmap0(neg_items, self.c)

        hyper_pos_dis = self.manifold.sqdist(hyper_users, hyper_pos_items, self.c)
        hyper_neg_dis = self.manifold.sqdist(hyper_users, hyper_neg_items, self.c)
        # hyper_pos_dis = hyp_distance(hyper_users, hyper_pos_items)
        # hyper_neg_dis = hyp_distance(hyper_users, hyper_neg_items)

        cf_loss = -1 * torch.mean(nn.LogSigmoid()
                                  (hyper_neg_dis - hyper_pos_dis))
        return cf_loss