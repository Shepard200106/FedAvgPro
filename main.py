import argparse, json
import datetime
import os
import logging
import torch, random

from server import *
from client import *
import models, datasets

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--no_models", type=int, help="Number of models")
    parser.add_argument("--type", type=str, help="Dataset type")
    parser.add_argument("--global_epochs", type=int, help="Global epochs")
    parser.add_argument("--local_epochs", type=int, help="Local epochs")
    parser.add_argument("--k", type=int, help="k value")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--momentum", type=float, help="Momentum")
    parser.add_argument("--lambda", type=float, help="Lambda value")
    parser.add_argument("--rate", type=float, help="Rate")
    parser.add_argument("--dp", type=bool, help="Dp")
    parser.add_argument("--C", type=float, help="C")
    parser.add_argument("--sigma", type=float, help="Sigma")
    parser.add_argument("--q", type=float, help="Q")
    parser.add_argument("--w", type=float, help="W")
    args = parser.parse_args()


    conf = vars(args)

    train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])

    server = Server(conf, eval_datasets)
    clients = []

    for c in range(conf["no_models"]):
        clients.append(Client(conf, server.global_model, train_datasets, c))

    print("\n\n")
    for e in range(conf["global_epochs"]):

        candidates = random.sample(clients, conf["k"])

        weight_accumulator = {}
        cnt={}

        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)
            cnt[name] = 0

        for c in candidates:
            diff = c.local_train(server.global_model)

            for name, params in server.global_model.state_dict().items():
                if name in diff:
                    weight_accumulator[name].add_(diff[name])
                    cnt[name]+=1

        server.model_aggregate(weight_accumulator,cnt)

        acc, loss = server.model_eval()

        print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))
