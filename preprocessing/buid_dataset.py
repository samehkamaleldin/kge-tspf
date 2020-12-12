import os
import gzip
import numpy as np
from tqdm import tqdm
import math


def export_entry(entry, out_fd):
    """

    :param entry:
    :param out_fd:
    :return:
    """
    for s, p, o in entry:
        out_fd.write("%s\t%s\t%s\n" % (s, p, o))


def export_quad_entry(entry, out_fd):
    """

    :param entry:
    :param out_fd:
    :return:
    """
    for s, p, o, c in entry:
        out_fd.write("%s\t%s\t%s\t%s\n" % (s, p, o, c))


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
    np.random.seed(seed)
    dataset_dir = "../data/dataset"
    kg_dir = "../data/kg"
    os.makedirs(dataset_dir) if not os.path.isdir(dataset_dir) else None

    train_all_fp = os.path.join(dataset_dir, "train.txt")
    train_all_fd = open(train_all_fp, "w")
    test_fp = os.path.join(dataset_dir, "test.txt")
    test_fd = open(test_fp, "w")

    train_ppi_fp = os.path.join(dataset_dir, "ppi_pos.txt")
    train_ppi_fd = open(train_ppi_fp, "w")

    train_data_ppi = []
    train_data_pos = []
    train_data_neg = []
    test_data_pos = []
    test_data_neg = []

    # get ppi
    print("= loading preprocessed OhmNet data")
    ppi_kg_filepath = os.path.join(kg_dir, "ppi_facts.txt.gz")
    ppi_kg_fd = gzip.open(ppi_kg_filepath, "rt")
    train_data_ppi.extend([l.strip().split() for l in ppi_kg_fd.readlines()])

    go_kg_filepath = os.path.join(kg_dir, "go_facts.txt.gz")
    go_kg_fd = gzip.open(go_kg_filepath, "rt")
    go_data = np.array([l.strip().split() for l in go_kg_fd.readlines()])
    go_tissue_list = list(np.unique(go_data[:, 1]))

    go_fact_dict_pos = {g: [] for g in go_tissue_list}
    go_fact_dict_neg = {g: [] for g in go_tissue_list}
    for gene, tissue, go_label, flag in tqdm(go_data, desc="grouping go annotation facts per tissue"):
        if flag == "1":
            go_fact_dict_pos[tissue].append([gene, tissue, go_label])
        else:
            go_fact_dict_neg[tissue].append([gene, tissue, go_label])

    # converting facts into numpy form
    for tissue in go_tissue_list:
        go_fact_dict_pos[tissue] = np.array(go_fact_dict_pos[tissue])
        go_fact_dict_neg[tissue] = np.array(go_fact_dict_neg[tissue])

    # shuffle facts per tissue
    for tissue in go_tissue_list:
        np.random.shuffle(go_fact_dict_pos[tissue])
        np.random.shuffle(go_fact_dict_neg[tissue])

    print("= processing tissue specific data")
    for tissue in go_tissue_list:
        data_splits_pos = [s for s in generate_random_splits(go_fact_dict_pos[tissue], nb_splits=10)]
        data_splits_neg = [s for s in generate_random_splits(go_fact_dict_neg[tissue], nb_splits=10)]
        test_data_pos.extend(data_splits_pos[0])
        test_data_neg.extend(data_splits_neg[0])
        for idx in range(1, 10):
            train_data_pos.extend(data_splits_pos[idx])
            train_data_neg.extend(data_splits_neg[idx])

    train_data_pos = [[s, p, o, "1"] for s, p, o in train_data_pos]
    train_data_neg = [[s, p, o, "0"] for s, p, o in train_data_neg]

    test_data_pos = [[s, p, o, "1"] for s, p, o in test_data_pos]
    test_data_neg = [[s, p, o, "0"] for s, p, o in test_data_neg]

    print("= Exporting dataset")
    export_entry(train_data_ppi, train_ppi_fd)

    export_quad_entry(train_data_pos, train_all_fd)
    export_quad_entry(train_data_neg, train_all_fd)

    export_quad_entry(test_data_pos, test_fd)
    export_quad_entry(test_data_neg, test_fd)

    train_ppi_fd.close()
    train_all_fd.close()
    test_fd.close()


if __name__ == '__main__':
    main()
