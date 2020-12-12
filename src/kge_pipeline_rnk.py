# -*- coding: utf-8 -*-

import math
import os
import gzip
import numpy as np
from tqdm import tqdm
from sklearn.pipeline import Pipeline

from libkge.embedding import TransE, ComplEx, DistMult, TriModel
from libkge import KgDataset

from sklearn.metrics import roc_auc_score, average_precision_score


def generate_random_splits(data, nb_splits=10):
    """
    split dataset into random equal size pieces

    :param data: np.array
        dataset np array
    :param nb_splits: int
        number of splits
    :return:
    """
    data_size = len(data)
    data_indices = np.arange(data_size)
    np.random.shuffle(data_indices)
    split_size = int(math.ceil(data_size/nb_splits))

    for idx in range(0, nb_splits):
        yield data[data_indices][idx * split_size:min(data_size, (idx + 1) * split_size), :]


def main():
    seed = 1234
    nb_epochs_then_check = None
    data_name = "TS-PROTEIN-GO"
    dataset_dir = "../data/dataset/"

    # loading dataset
    train_fp = os.path.join(dataset_dir, "train.txt.gz")
    train_facts_labeled = [l.strip().split("\t") for l in gzip.open(train_fp, "rt").readlines()]
    train_facts = np.array([[s, p, o] for s, p, o, f in train_facts_labeled if f == "1"])
    # train_facts_neg = np.array([[s, p, o] for s, p, o, f in train_facts_labeled if f == "0"])

    test_fp = os.path.join(dataset_dir, "test.txt.gz")
    test_facts_labeled = [l.strip().split("\t") for l in gzip.open(test_fp, "rt").readlines()]
    test_facts = np.array([[s, p, o] for s, p, o, f in test_facts_labeled if f == "1"])
    test_facts_neg = np.array([[s, p, o] for s, p, o, f in test_facts_labeled if f == "0"])

    tissue_list = list(set(list(test_facts[:, 1])))
    dataset = KgDataset(name=data_name)
    dataset.load_triples(train_facts, "train")
    dataset.load_triples(test_facts, "test")
    dataset.load_triples(test_facts_neg, "test_neg")

    del train_facts
    del test_facts
    del test_facts_neg

    train_data = dataset.data["train"]
    test_data = dataset.data["test"]
    test_data_neg = dataset.data["test_neg"]
    tissue_list = dataset.get_rel_indices(tissue_list)

    # model pipeline definition
    model = TriModel(seed=seed, loss="pt_log", verbose=2)
    pipe_model = Pipeline([
        ('kge_model', model)
    ])

    # set model parameters
    model_params = {
        'kge_model__em_size': 30,
        'kge_model__lr': 0.01,
        'kge_model__nb_negs': 2,
        'kge_model__nb_epochs': 200,
        'kge_model__batch_size': 4000,
        'kge_model__nb_ents': dataset.get_ents_count(),
        'kge_model__nb_rels': dataset.get_rels_count()
    }

    # add parameters to the model then call fit method
    pipe_model.set_params(**model_params)
    pipe_model.fit(X=train_data)

    ts_auc_roc_list = []
    ts_auc_pr_list = []
    print("============================================================")
    print("= Tissue-specific evaluation                               =")
    print("============================================================")
    for tissue_idx in tissue_list:
        tissue_name = dataset.get_rel_labels([tissue_idx])[0]

        ts_test_facts_pos = np.array([[s, p, o] for s, p, o in test_data if p == tissue_idx])
        ts_test_facts_neg = np.array([[s, p, o] for s, p, o in test_data_neg if p == tissue_idx])
        set_test_facts_all = np.concatenate([ts_test_facts_pos, ts_test_facts_neg])
        se_test_facts_labels = np.concatenate([np.ones([len(ts_test_facts_pos)]), np.zeros([len(ts_test_facts_neg)])])
        se_test_facts_scores = model.predict(set_test_facts_all)

        se_auc_pr = average_precision_score(se_test_facts_labels, se_test_facts_scores)
        se_auc_roc = roc_auc_score(se_test_facts_labels, se_test_facts_scores)

        ts_auc_roc_list.append(se_auc_roc)
        ts_auc_pr_list.append(se_auc_pr)

        print("= AUC-ROC: %1.4f - AUC-PR: %1.4f > %s" % (se_auc_roc, se_auc_pr, tissue_name), flush=True)

    se_auc_roc_list_avg = np.average(ts_auc_roc_list)
    se_auc_pr_list_avg = np.average(ts_auc_pr_list)

    print("============================================================")
    print("= AUC-ROC: %1.4f - AUC-PR: %1.4f > [AVERAGE]" % (se_auc_roc_list_avg, se_auc_pr_list_avg), flush=True)
    print("============================================================")


if __name__ == '__main__':
    main()

