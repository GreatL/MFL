"""Base model class."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import manifolds
import models.encoders as encoders
from models.decoders import model2decoder
from utils.eval_utils import acc_f1
from .hyper_nets import LorentzLinear
from onmt.manifolds.lorentz import Lorentz


class MyLinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(MyLinearLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = F.linear(x, self.weight)
        return x

class BaseModel(nn.Module):
    """
    Base model for MFL.
    """
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold 
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.])) 
        self.manifold = getattr(manifolds, self.manifold_name)() 
        if self.manifold.name in ['Lorentz', 'Hyperboloid']: 
            args.feat_dim = args.feat_dim + 1 
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args) 
        self.relu = nn.ReLU() 
        
        FEATURE_DIM = args.feature_dim 
        HIDDEN_SIZE = args.hidden_size 
        self.attention = nn.MultiheadAttention(embed_dim=HIDDEN_SIZE, num_heads=8) 
        self.text_weight = MyLinearLayer(FEATURE_DIM, HIDDEN_SIZE) 
        self.image_weight = MyLinearLayer(FEATURE_DIM, HIDDEN_SIZE) 
        self.text_liner = LorentzLinear(FEATURE_DIM + 1, HIDDEN_SIZE) 
        self.image_liner = LorentzLinear(FEATURE_DIM + 1, HIDDEN_SIZE) 
        self.f = LorentzLinear(HIDDEN_SIZE + 1, args.dim + 1) 
        self.f_ML = LorentzLinear(args.dim, args.dim) 

    def encode(self, text_vector, image_vector, adj): 
        o_text = torch.zeros_like(text_vector)
        text_vector = torch.cat([o_text[:, 0:1], text_vector], dim=1)
        text_vector = self.relu(self.text_liner(text_vector))
        o_image = torch.zeros_like(image_vector)
        image_vector = torch.cat([o_image[:, 0:1], image_vector], dim=1)
        image_vector = self.relu(self.image_liner(image_vector))
        text_vector_a = text_vector.unsqueeze(1)  
        image_vector_a = image_vector.unsqueeze(1)  
        self.attention_weights = []
        attention_output, attention_weights = self.attention(text_vector_a.permute(1, 0, 2), image_vector_a.permute(1, 0, 2), image_vector_a.permute(1, 0, 2))
       
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            o_text = torch.zeros_like(text_vector)
            text_vector = torch.cat([o_text[:, 0:1], text_vector], dim=1)
            text_vector = self.manifold.expmap0(text_vector)
            o_image = torch.zeros_like(image_vector)
            image_vector = torch.cat([o_image[:, 0:1], image_vector], dim=1)
            image_vector = self.manifold.expmap0(image_vector)
            o_attention = torch.zeros(attention_output.size(0), attention_output.size(1), 1, device=attention_output.device)  
            attention_output = torch.cat([o_attention, attention_output], dim=2)  
            attention_output = self.manifold.expmap0(attention_output)  

        X = torch.stack([text_vector, image_vector], dim=0)
        m_tv = Lorentz().mid_point(X)  
        fusion_vector = Lorentz().mid_point(torch.stack([m_tv, attention_output[0]], dim=0)) 
        x = self.f(fusion_vector)  
        h = self.encoder.encode(x, adj)
        h = self.f_ML(h)
        self.last_attention_weights = attention_weights 
        return h

    def compute_metrics(self, embeddings, data, split, lamda):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError

class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)
        self.relu = nn.ReLU()

        FEATURE_DIM = args.feature_dim
        HIDDEN_SIZE = args.hidden_size
        self.text_weight = MyLinearLayer(FEATURE_DIM, HIDDEN_SIZE)
        self.image_weight = MyLinearLayer(FEATURE_DIM, HIDDEN_SIZE)
        self.text_liner = LorentzLinear(HIDDEN_SIZE + 1, HIDDEN_SIZE)
        self.image_liner = LorentzLinear(HIDDEN_SIZE + 1, HIDDEN_SIZE)
        self.f = LorentzLinear(HIDDEN_SIZE + 1, args.dim + 1)
        self.f_ML = LorentzLinear(args.dim, args.dim)

    def encode(self, text_vector, image_vector, adj):
        text_vector = self.relu(self.text_weight(text_vector))
        image_vector = self.relu(self.image_weight(image_vector))
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            o_text = torch.zeros_like(text_vector)
            text_vector = torch.cat([o_text[:, 0:1], text_vector], dim=1)
            o_image = torch.zeros_like(image_vector)
            image_vector = torch.cat([o_image[:, 0:1], image_vector], dim=1)
            if self.manifold.name == 'Lorentz':
                text_vector = self.manifold.expmap0(text_vector)
                image_vector = self.manifold.expmap0(image_vector)
        X = torch.stack([text_vector, image_vector], dim=0)
        fusion_vector = Lorentz().mid_point(X)
        x = self.f(fusion_vector)
        h = self.encoder.encode(x, adj)
        h = self.f_ML(h)
        return h

    def compute_metrics(self, embeddings, data, split, lamda):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError

class NCModel(BaseModel):
    """
    Base model for node classification task.
    tangent version
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return output[idx]

    def get_embedding(self, h, adj):
        final_embed, _ = self.decoder.decode(h, adj)
        return final_embed

    def compute_metrics(self, embeddings, data, split, lamda):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.cross_entropy(output, data['labels'][idx], self.weights)
        if lamda != 0:
            loss = loss + abs(lamda * self.compute_hyp_loss(embeddings[idx]))
        acc, f1, classification_report = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics, classification_report

    def compute_hyp_loss(self, embeddings):
        z_e = Lorentz().logmap0(embeddings)
        z_c = torch.sum(z_e, dim=0) / len(z_e)
        z = z_e - z_c
        sum = 0
        for feature in z:
            sum += torch.norm(feature)
        z_hdo = sum / len(z)
        tanh = nn.Tanh()
        loss_hyp = tanh(-z_hdo)
        return loss_hyp

    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]
