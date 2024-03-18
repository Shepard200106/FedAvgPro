# coding: utf-8
import torch
import numpy as np


def add_noise(parameters, dp_type, args):
    if dp_type == 0:
        return parameters
    elif dp_type == 1:
        noise = torch.tensor(np.random.laplace(0, args.sigma, parameters.shape)).to(args.device)
    else:
        noise = torch.FloatTensor(parameters.shape).normal_(0, args.sigma).to(args.device)
    return parameters.add_(noise)