import argparse
import time
import numpy as np
import networkx as nx
import torch as th
import pandas as pd
import seaborn as sb
import os
import sys
import scipy.sparse as sp
import pickle as pkl
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import time

from models.gcn import GCN

# from models.gcn_mp import GCN
# from models.gcn_spmv import GCN

results = []
times = []
edges_count = []
nodes_count = []


def get_labels(base):
    f = open('{}_labels.txt'.format(base)).read()

    labels = []

    labels_in_file = f.split(',')

    for split in labels_in_file:

        if split.count('[') > 0:
            split = split.replace('[', '')

        if split.count('\'') > 0:
            split = split.replace('\'', '')

        if split.count(']') > 0:
            split = split.replace(']', '')

        labels.append(split)

    return labels


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def _normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.asarray(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def cm2df(cm, labels):
    df = pd.DataFrame()
    # rows
    for i, row_label in enumerate(labels):
        rowdata = {}
        # columns
        for j, col_label in enumerate(labels):
            rowdata[col_label] = cm[i, j]
        df = df.append(pd.DataFrame.from_dict({row_label: rowdata}, orient='index'))
    return df[labels]


def load_data(dataset, model, step, fold):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        if names[i] == 'graph':
            with open("data/{}/ind.mine_{}_conj_{}_step_{}.{}".format(dataset, model, fold, step, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))
        else:
            with open("data/{}/ind.mine_{}_conj_{}_step_{}.{}".format(dataset, model, fold, step, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/{}/ind.mine_{}_conj_{}_step_{}.test.index".format(dataset, model, fold, step))
    test_idx_range = np.sort(test_idx_reorder)

    class_num = len(ally[0])
    print('Class num', class_num)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    size = features.shape[1]
    print('Model Feature extracted size', size)

    u, v = [], []

    for node in range(len(graph)):

        source = node

        connections = graph[node]

        for target in connections:
            u.append(source)
            v.append(target)

    adj = dgl.graph((u, v), num_nodes=len(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(ally))
    idx_val = range(len(ally))

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    ft_norm = _normalize(features)

    features = np.asarray(ft_norm.todense())

    labels = np.where(labels)[1]

    features = th.FloatTensor(features)
    labels = th.LongTensor(labels)
    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)
    test_mask = th.BoolTensor(test_mask)

    return adj, features, labels, train_mask, val_mask, test_mask, size, class_num


def evaluate(model, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def class_evaluate(model, features, labels, mask, base):
    model.eval()
    with th.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        cm = confusion_matrix(labels.cpu().numpy(), indices.cpu().numpy())
        cm = cm2df(cm, get_labels(base))
        classification = classification_report(labels.cpu().numpy(), indices.cpu().numpy(),
                                               target_names=get_labels(base))
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels), classification, cm


def main(args):
    best_model = None
    # load and preprocess dataset
    graph, features, labels, train_mask, val_mask, test_mask, size, num_classes = load_data(args.images, args.model,
                                                                                            args.walks, args.fold)
    in_feats = size
    n_classes = num_classes

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        th.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    g = graph
    g = g.to(0)
    n_edges = g.number_of_edges()
    n_nodes = g.number_of_nodes()

    print("""----Data statistics------'
          #Classes %d
          #Train samples %d
          #Val samples %d
          #Test samples %d
          #Edges from graph %d
          #Nodes from graph %d """ %
          (n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item(), n_edges, n_nodes))

    # normalization
    degs = g.in_degrees().float()
    norm = th.pow(degs, -0.5)
    norm[th.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    if args.walks == 0:
        model_name = 'saved_models/{}/{}_SAVED_MODEL_FULL_CONNECTED_{}.pth'.format(args.model ,args.images, args.model)
    elif args.walks > 0 and args.walks <= 10:
        model_name = 'saved_models/{}/{}_SAVED_MODEL_RANDOMWALK_{}_{}.pth'.format(args.model, args.images, args.model, args.walks)
    elif args.walks > 10 and args.walks <= 20:
        model_name = 'saved_models/{}/{}_SAVED_MODEL_RANDOMCUT_{}_{}.pth'.format(args.model, args.images, args.model, args.walks)
    elif args.walks > 20 and args.walks <= 30:
        model_name = 'saved_models/{}/{}_SAVED_MODEL_RANDOMWEIGHTED_{}_{}.pth'.format(args.model, args.images, args.model, args.walks)
    elif args.walks > 30 and args.walks <= 40:
        model_name = 'saved_models/{}/{}_SAVED_MODEL_RANDOMEDGE_CREATION_{}_{}.pth'.format(args.model, args.images, args.model, args.walks)

    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout)
    # try:
    #    model.load_state_dict(th.load(model_name))
    #    print('MODELO CARREGADO COM SUCESSO')
    # except:
    #    print('MODELO CRIADO COM SUCESSO')

    if cuda:
        model.cuda()
    loss_fcn = th.nn.CrossEntropyLoss()

    optimizer = th.optim.Adam(model.parameters(),
                              lr=args.lr,
                              weight_decay=args.weight_decay)

    if args.walks == 0:
        path = 'logs/{}/STD/LOGS_KFOLD/FullConnected_KFOLD_{}_STEP_{}_LR_{}_DROP_{}_ARC_{}_NEURON_{}.txt'.format(
            args.images, str(args.fold), str(args.walks), str(args.lr), str(args.dropout),
            args.model, args.n_hidden
        )
    elif args.walks > 0 and args.walks <= 10:
        path = 'logs/{}/STD/LOGS_KFOLD/Random_Walk_KFOLD_{}_STEP_{}_LR_{}_DROP_{}_ARC_{}_NEURON_{}.txt'.format(
            args.images, str(args.fold), str(args.walks), str(args.lr), str(args.dropout),
            args.model, args.n_hidden
        )
    elif args.walks > 10 and args.walks <= 20:
        path = 'logs/{}/STD/LOGS_KFOLD/Random_Cut_KFOLD_{}_STEP_{}_LR_{}_DROP_{}_ARC_{}_NEURON_{}.txt'.format(
            args.images, str(args.fold), str(args.walks), str(args.lr), str(args.dropout),
            args.model, args.n_hidden
        )

    elif args.walks > 20 and args.walks <= 30:
        path = 'logs/{}/STD/LOGS_KFOLD/Random_Weighted_KFOLD_{}_STEP_{}_LR_{}_DROP_{}_ARC_{}_NEURON_{}.txt'.format(
            args.images, str(args.fold), str(args.walks), str(args.lr), str(args.dropout),
            args.model, args.n_hidden
        )
    
    elif args.walks > 30 and args.walks <= 40:
        path = 'logs/{}/STD/LOGS_KFOLD/Random_EDGE_CREATION_KFOLD_{}_STEP_{}_LR_{}_DROP_{}_ARC_{}_NEURON_{}.txt'.format(
            args.images, str(args.fold), str(args.walks), str(args.lr), str(args.dropout),
            args.model, args.n_hidden
        )

    file = open(path, 'w')
    file.write('NUMBER OF EDGES = ' + str(n_edges) + '\n' + '\n')
    time_start = time.time()
    model.train()
    patience = args.early
    for epoch in range(args.n_epochs):
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        loss_test = loss_fcn(logits[test_mask], labels[test_mask])
        if epoch == 0:
            best_test_loss = loss_test
        if loss_test < best_test_loss:
            best_model = model
            patience = args.early
        if patience == 0:
            print("Stop by Early Stopping...")
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = evaluate(model, features, labels, train_mask)
        test_acc = evaluate(model, features, labels, test_mask)
        print("Epoch {:05d} | Loss {:.8f} | Train_Accuracy {:.9f} | Test_Accuracy {:.9f} | Test_Loss {:.9f}".format(epoch, loss.item(),
                                                                                                   acc, test_acc, loss_test.item()))
        file.write(
            'Epoch ' + str(epoch) + ' | Loss ' + str(loss.item()) + ' | ' + 'Train_Accuracy ' + str(acc) + ' | \n')
        patience-=1
        
    time_stop = time.time()
    process_time = time_stop - time_start
    times.append(process_time)
    edges_count.append(n_edges)
    nodes_count.append(n_nodes)
    file.write('Process time execution = ' + str(process_time) + '\n')
    print()

    test_acc, classification, cm = class_evaluate(best_model, features, labels, test_mask, args.images)
    file.write(str(test_acc) + '\n')
    file.write(str(classification) + '\n')
    cm.to_csv(path.replace('.txt', '.csv'))
    results.append(test_acc)
    print("Test accuracy {:.2%}".format(test_acc))
    if args.walks == 0:
        th.save(best_model, 'saved_models/{}/full_connected_model_{}_{}_{}.pth'.format(args.model, args.images, args.model, args.walks))
        th.save(best_model.state_dict(), model_name)

        th.save(best_model, 'saved_models/{}/full_connected_model_{}.pth'.format(args.model, args.model))
        model_name = 'saved_models/{}/{}_SAVED_MODEL_FULL_CONNECTED_{}.pth'.format(args.model, args.images, args.model)
        th.save(best_model.state_dict(), model_name)

    elif args.walks > 0 and args.walks <= 10:

        th.save(best_model, 'saved_models/{}/random_walk_model_{}_{}_{}.pth'.format(args.model, args.images, args.model, args.walks))
        th.save(best_model.state_dict(), model_name)

        th.save(best_model, 'saved_models/{}/random_walk_model_{}_{}.pth'.format(args.model, args.images, args.model))
        model_name = 'saved_models/{}/{}_SAVED_MODEL_RANDOMWALK_{}.pth'.format(args.model, args.images, args.model)
        th.save(best_model.state_dict(), model_name)

    elif args.walks > 10 and args.walks <= 20:
        th.save(best_model, 'saved_models/{}/random_cut_model_{}_{}_{}.pth'.format(args.model, args.images, args.model, args.walks))
        th.save(best_model.state_dict(), model_name)

        th.save(best_model, 'saved_models/{}/random_cut_model_{}_{}.pth'.format(args.model, args.images, args.model))
        model_name = 'saved_models/{}/{}_SAVED_MODEL_RANDOMCUT_{}.pth'.format(args.model, args.images, args.model)
        th.save(best_model.state_dict(), model_name)

    elif args.walks > 20 and args.walks <= 30:
        th.save(best_model, 'saved_models/{}/random_weighted_model_{}_{}_{}.pth'.format(args.model, args.images, args.model, args.walks))
        th.save(best_model.state_dict(), model_name)

        th.save(best_model, 'saved_models/{}/random_weighted_model_{}_{}.pth'.format(args.model, args.images, args.model))
        model_name = 'saved_models/{}/{}_SAVED_MODEL_RANDOMWEIGHTED_{}.pth'.format(args.model, args.images, args.model)
        th.save(best_model.state_dict(), model_name)

    elif args.walks > 30 and args.walks <= 40:
        th.save(best_model, 'saved_models/{}/random_edge_model_{}_{}_{}.pth'.format(args.model, args.images, args.model, args.walks))
        th.save(best_model.state_dict(), model_name)

        th.save(best_model, 'saved_models/{}/random_edge_model_{}_{}.pth'.format(args.model, args.images, args.model))
        model_name = 'saved_models/{}/{}_SAVED_MODEL_RANDOMEDGE_{}.pth'.format(args.model, args.images, args.model)
        th.save(best_model.state_dict(), model_name)

    print('MODELO SALVO COM SUCESSO!!')


if __name__ == '__main__':

    walks = [i for i in range(41)]

    models = ['VGG16']

    lrs = [0.01, 0.05, 0.001, 0.005]

    dropouts = [0.3, 0.5, 0.9]

    neurons = [256, 512]

    conjuntos = [i for i in range(5)]

    datasets = ['UNREL']

    for dataset in datasets:

        for model in models:

            for neuron in neurons:

                for lr in lrs:

                    for dropout in dropouts:

                        for walk in walks:

                            for conjunto in conjuntos:
                                parser = argparse.ArgumentParser(description='GCN')
                                register_data_args(parser)
                                parser.add_argument("--dropout", type=float, default=dropout,
                                                    help="dropout probability")
                                parser.add_argument("--gpu", type=int, default=0,
                                                    help="gpu")
                                parser.add_argument("--walks", type=int, default=walk,
                                                    help="random walk steps")
                                parser.add_argument("--lr", type=float, default=lr,
                                                    help="learning rate")
                                parser.add_argument("--early", type=int, default=500,
                                                    help="early stoping, trying to avoid overfitting")
                                parser.add_argument("--n-epochs", type=int, default=2000,
                                                    help="number of training epochs")
                                parser.add_argument("--n-hidden", type=int, default=neuron,
                                                    help="number of hidden gcn units")
                                parser.add_argument("--n-layers", type=int, default=2,
                                                    help="number of hidden gcn layers")
                                parser.add_argument("--weight-decay", type=float, default=0.000005,
                                                    help="Weight for L2 loss")
                                parser.add_argument("--self-loop", action='store_true',
                                                    help="graph self-loop (default=False)")
                                parser.add_argument("--images", type=str, default=dataset,
                                                    help="Dataset that will be used")
                                parser.add_argument("--model", type=str, default=model,
                                                    help="Model that will be used")
                                parser.add_argument("--fold", type=str, default=conjunto,
                                                    help="Fold that will be used")
                                parser.set_defaults(self_loop=False)
                                args = parser.parse_args()

                                print(args)

                                main(args)

                            if walk == 0:
                                std_file_name = 'logs/{}/STD/{}_Full_Connected_{}_{}_{}_{}.txt'.format(dataset, dataset,
                                                                                                       lr, dropout,
                                                                                                       model, neuron)
                                file = open(std_file_name, 'w')
                                file.write('Standard deviation of Full Connected ' + str(np.std(results)) + '\n')
                                file.write('Mean between the experiments is ' + str(np.mean(results)) + '\n')
                                file.write('Time of process [Mean] is ' + str(np.mean(times)) + '\n')
                                file.write('Number of nodes used is ' + str(np.mean(nodes_count)) + '\n')
                                file.write('Number of edges is ' + str(np.mean(edges_count)) + '\n')
                                a = 0
                                for values in results:
                                    file.write('Conjunto ' + str(a) + ' = ' + str(values) + '\n')
                                    a = a + 1
                                file.close()
                            elif walk > 0 and walk <= 10:
                                std_file_name = 'logs/{}/STD/{}_RandomWalk_{}_{}_{}_{}_{}.txt'.format(dataset, dataset,
                                                                                                      walk, lr, dropout,
                                                                                                      model, neuron)
                                file = open(std_file_name, 'w')
                                file.write('Standard deviation of Random Walk ' + str(np.std(results)) + '\n')
                                file.write('Mean between the experiments is ' + str(np.mean(results)) + '\n')
                                file.write('Time of process [Mean] is ' + str(np.mean(times)) + '\n')
                                file.write('Number of nodes used is ' + str(np.mean(nodes_count)) + '\n')
                                file.write('Number of edges is ' + str(np.mean(edges_count)) + '\n')
                                a = 0
                                for values in results:
                                    file.write('Conjunto ' + str(a) + ' = ' + str(values) + '\n')
                                    a = a + 1
                                file.close()

                            elif walk > 10 and walk <= 20:
                                std_file_name = 'logs/{}/STD/{}RandomCut_{}_{}_{}_{}_{}.txt'.format(dataset, dataset,
                                                                                                    walk, lr, dropout,
                                                                                                    model, neuron)
                                file = open(std_file_name, 'w')
                                file.write('Standard deviation of Random Cut ' + str(np.std(results)) + '\n')
                                file.write('Mean between the experiments is ' + str(np.mean(results)) + '\n')
                                file.write('Time of process [Mean] is ' + str(np.mean(times)) + '\n')
                                file.write('Number of nodes used is ' + str(np.mean(nodes_count)) + '\n')
                                file.write('Number of edges is ' + str(np.mean(edges_count)) + '\n')
                                a = 0
                                for values in results:
                                    file.write('Conjunto ' + str(a) + ' = ' + str(values) + '\n')
                                    a = a + 1
                                file.close()
                            elif walk > 20 and walk <= 30:
                                std_file_name = 'logs/{}/STD/{}RandomWighted_{}_{}_{}_{}_{}.txt'.format(dataset, dataset,
                                                                                                    walk, lr, dropout,
                                                                                                    model, neuron)
                                file = open(std_file_name, 'w')
                                file.write('Standard deviation of Random Weighted ' + str(np.std(results)) + '\n')
                                file.write('Mean between the experiments is ' + str(np.mean(results)) + '\n')
                                file.write('Time of process [Mean] is ' + str(np.mean(times)) + '\n')
                                file.write('Number of nodes used is ' + str(np.mean(nodes_count)) + '\n')
                                file.write('Number of edges is ' + str(np.mean(edges_count)) + '\n')
                                a = 0
                                for values in results:
                                    file.write('Conjunto ' + str(a) + ' = ' + str(values) + '\n')
                                    a = a + 1
                                file.close()
                            elif walk > 30 and walk <= 40:
                                std_file_name = 'logs/{}/STD/{}RandomEdge_{}_{}_{}_{}_{}.txt'.format(dataset, dataset,
                                                                                                    walk, lr, dropout,
                                                                                                    model, neuron)
                                file = open(std_file_name, 'w')
                                file.write('Standard deviation of Random Edge ' + str(np.std(results)) + '\n')
                                file.write('Mean between the experiments is ' + str(np.mean(results)) + '\n')
                                file.write('Time of process [Mean] is ' + str(np.mean(times)) + '\n')
                                file.write('Number of nodes used is ' + str(np.mean(nodes_count)) + '\n')
                                file.write('Number of edges is ' + str(np.mean(edges_count)) + '\n')
                                a = 0
                                for values in results:
                                    file.write('Conjunto ' + str(a) + ' = ' + str(values) + '\n')
                                    a = a + 1
                                file.close()
                            results = []
                            edges_count = []
                            nodes_count = []
                            times = []
