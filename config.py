import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'lr': (0.0003, 'learning rate'),
        'dropout': (0.0, 'dropout probability'),
        'cuda': (1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (5000, 'maximum number of epochs to train for'),
        'weight-decay': (0., 'l2 regularization strength'),
        'optimizer': ('radam', 'which optimizer to use, can be any of [rsgd, radam, Adam]'),
        'momentum': (0.999, 'momentum in optimizer'),
        'patience': (500, 'patience for early stopping'),
        'seed': (2024, 'seed for training'),
        'log-freq': (5, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'save': (1, '1 to save model and logs and 0 otherwise'),
        'save-dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        'sweep-c': (0, ''),
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        'gamma': (0.5, 'gamma for lr scheduler'),
        'print-epoch': (True, ''),
        'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping'),
        'min-epochs': (100, 'do not early stop before min-epochs')
    },
    'model_config': {
        'model': ('HyboNet', 'HyboNet encoder or GCN'),
        'dim': (32, 'embedding dimension'),
        'manifold': ('Lorentz', 'Lorentz manifold'),
        'c': (None, 'hyperbolic radius, set to None for trainable curvature'),
        'r': (2., 'fermi-dirac decoder parameter'),
        't': (1., 'fermi-dirac decoder parameter'),
        'lamda': (0.1, 'scale factor'),
        'hidden_size': (256, 'hidden size'),
        'pretrained-embeddings': (None, 'path to pretrained embeddings (.npy file) for Shallow node classification'),
        'pos-weight': (0, 'whether to upweight positive class in node classification tasks'),
        'num-layers': (3, 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('None', 'which activation function to use (or None for no activation)'),
        'double-precision': ('0', 'whether to use double precision'),
        'use_att': (1, 'whether to use hyperbolic attention or not'),
        'local-agg': (1, 'whether to local tangent space aggregation or not')
    },
    'data_config': {
        'dataset': ('pheme', 'which dataset to use'),
        'feature_dim': (768, 'feature dimension'),
        'use-feats': (1, 'whether to use node features or not'),
        'normalize-feats': (1, 'whether to normalize input node features'),
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix'),
        'split-seed': (1234, 'seed for data splits (train/test/val)'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
