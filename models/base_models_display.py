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
        print("text_vector",text_vector.shape)
        print("image_vector",text_vector.shape)
        if self.manifold.name in ['Lorentz', 'Hyperboloid']:
            o_text = torch.zeros_like(text_vector)
            text_vector = torch.cat([o_text[:, 0:1], text_vector], dim=1)
            o_image = torch.zeros_like(image_vector)
            image_vector = torch.cat([o_image[:, 0:1], image_vector], dim=1)
            if self.manifold.name == 'Lorentz':
                text_vector = self.manifold.expmap0(text_vector)
                image_vector = self.manifold.expmap0(image_vector)
        print("text_vector",text_vector.shape)
        print("image_vector",text_vector.shape)
        X = torch.stack([text_vector, image_vector], dim=0)
        print("X",X.shape)
        fusion_vector = Lorentz().mid_point(X)
        print("fusion_vector",fusion_vector.shape)
        x = self.f(fusion_vector)
        print("x",x.shape)
        h = self.encoder.encode(x, adj)
        print("h",h.shape)
        h = self.f_ML(h)
        print("h",h.shape)
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
            loss = abs(loss + (lamda * self.compute_hyp_loss(embeddings[idx])))    
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
