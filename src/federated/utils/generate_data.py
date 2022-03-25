#!/usr/bin/env python3
from examples.constants import GENERATE_DATA_DESC, NUM_PARTIES_DESC, DATASET_DESC, PATH_DESC, PER_PARTY, \
    STRATIFY_DESC, FL_DATASETS, NEW_DESC, PER_PARTY_ERR, NAME_DESC
from ibmfl.util.datasets import load_nursery, load_mnist, load_adult, load_compas, load_german, \
    load_higgs, load_airline, load_diabetes, load_binovf, load_multovf, load_linovf, \
    load_simulated_federated_clustering, load_leaf_femnist, load_cifar10
import os
import sys
import csv
import time
import argparse
import numpy as np

fl_path = os.path.abspath('.')
if fl_path not in sys.path:
    sys.path.append(fl_path)


def setup_parser():
    """
    Sets up the parser for Python script

    :return: a command line parser
    :rtype: argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(description=GENERATE_DATA_DESC)
    p.add_argument("--num_parties", "-n", help=NUM_PARTIES_DESC,
                   type=int, required=True)
    p.add_argument("--dataset", "-d", choices=FL_DATASETS,
                   help=DATASET_DESC, required=True)
    p.add_argument("--data_path", "-p", help=PATH_DESC)
    p.add_argument("--points_per_party", "-pp", help=PER_PARTY,
                   nargs="+", type=int, required=True)
    p.add_argument("--stratify", "-s", help=STRATIFY_DESC, action="store_true")
    p.add_argument("--create_new", "-new", action="store_true", help=NEW_DESC)
    p.add_argument("--name", help=NAME_DESC)
    return p


def save_cifar10_party_data(nb_dp_per_party, should_stratify, party_folder, dataset_folder):
    """
    Saves Cifar10 party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    :param dataset_foler: folder to save dataset
    :type dataset_folder: `str`
    """
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    labels, train_counts = np.unique(y_train, return_counts=True)
    te_labels, test_counts = np.unique(y_test, return_counts=True)
    if np.all(np.isin(labels, te_labels)):
        print("Warning: test set and train set contain different labels")

    num_train = np.shape(y_train)[0]
    num_test = np.shape(y_test)[0]
    num_labels = np.shape(np.unique(y_test))[0]
    nb_parties = len(nb_dp_per_party)

    if should_stratify:
        # Sample according to source label distribution
        train_probs = {
            label: train_counts[label] / float(num_train) for label in labels}
        test_probs = {label: test_counts[label] /
                      float(num_test) for label in te_labels}
    else:
        # Sample uniformly
        train_probs = {label: 1.0 / len(labels) for label in labels}
        test_probs = {label: 1.0 / len(te_labels) for label in te_labels}
    for idx, dp in enumerate(nb_dp_per_party):
        train_p = np.array([train_probs[y_train[idx]]
                            for idx in range(num_train)])
        train_p = np.array(train_p)
        train_p /= np.sum(train_p)
        train_indices = np.random.choice(num_train, dp, p=train_p)
        test_p = np.array([test_probs[y_test[idx]] for idx in range(num_test)])
        test_p /= np.sum(test_p)

        # Split test evenly
        test_indices = np.random.choice(
            num_test, int(num_test / nb_parties), p=test_p)

        x_train_pi = x_train[train_indices]
        y_train_pi = y_train[train_indices]
        x_test_pi = x_test[test_indices]
        y_test_pi = y_test[test_indices]

        # Now put it all in an npz
        name_file = 'data_party' + str(idx) + '.npz'
        name_file = os.path.join(party_folder, name_file)
        np.savez(name_file, x_train=x_train_pi, y_train=y_train_pi,
                 x_test=x_test_pi, y_test=y_test_pi)

        print_statistics(idx, x_test_pi, x_train_pi, num_labels, y_train_pi)

        print('Finished! :) Data saved in ', party_folder)


def save_mnist_party_data(nb_dp_per_party, should_stratify, party_folder, dataset_folder):
    """
    Saves MNIST party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    :param dataset_folder: folder to save dataset
    :type data_path: `str`
    :param dataset_foler: folder to save dataset
    :type dataset_folder: `str`
    """
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    (x_train, y_train), (x_test, y_test) = load_mnist(download_dir=dataset_folder)
    labels, train_counts = np.unique(y_train, return_counts=True)
    te_labels, test_counts = np.unique(y_test, return_counts=True)
    if np.all(np.isin(labels, te_labels)):
        print("Warning: test set and train set contain different labels")

    num_train = np.shape(y_train)[0]
    num_test = np.shape(y_test)[0]
    num_labels = np.shape(np.unique(y_test))[0]
    nb_parties = len(nb_dp_per_party)

    if should_stratify:
        # Sample according to source label distribution
        train_probs = {
            label: train_counts[label] / float(num_train) for label in labels}
        test_probs = {label: test_counts[label] /
                      float(num_test) for label in te_labels}
    else:
        # Sample uniformly
        train_probs = {label: 1.0 / len(labels) for label in labels}
        test_probs = {label: 1.0 / len(te_labels) for label in te_labels}

    for idx, dp in enumerate(nb_dp_per_party):
        train_p = np.array([train_probs[y_train[idx]]
                            for idx in range(num_train)])
        train_p /= np.sum(train_p)
        train_indices = np.random.choice(num_train, dp, p=train_p)
        test_p = np.array([test_probs[y_test[idx]] for idx in range(num_test)])
        test_p /= np.sum(test_p)

        # Split test evenly
        test_indices = np.random.choice(
            num_test, int(num_test / nb_parties), p=test_p)

        x_train_pi = x_train[train_indices]
        y_train_pi = y_train[train_indices]
        x_test_pi = x_test[test_indices]
        y_test_pi = y_test[test_indices]

        # Now put it all in an npz
        name_file = 'data_party' + str(idx) + '.npz'
        name_file = os.path.join(party_folder, name_file)
        np.savez(name_file, x_train=x_train_pi, y_train=y_train_pi,
                 x_test=x_test_pi, y_test=y_test_pi)

        print_statistics(idx, x_test_pi, x_train_pi, num_labels, y_train_pi)

        print('Finished! :) Data saved in ', party_folder)


if __name__ == '__main__':
    # Parse command line options
    parser = setup_parser()
    args = parser.parse_args()

    # Collect arguments
    num_parties = args.num_parties
    dataset = args.dataset
    data_path = args.data_path
    points_per_party = args.points_per_party
    stratify = args.stratify
    create_new = args.create_new
    exp_name = args.name

    # Check for errors
    if len(points_per_party) == 1:
        points_per_party = [points_per_party[0] for _ in range(num_parties)]
    elif len(points_per_party) != num_parties:
        parser.error(PER_PARTY_ERR)

    if data_path is not None:
        if not os.path.exists(data_path):
            print('Data Path:{} does not exist.'.format(data_path))
            print('Creating {}'.format(data_path))
            try:
                os.makedirs(data_path, exist_ok=True)
            except OSError:
                print('Creating directory {} failed'.format(data_path))
                sys.exit(1)
        folder_party_data = os.path.join(data_path, "data")
        folder_dataset = os.path.join(data_path, "datasets")
    else:
        folder_party_data = os.path.join("examples", "data")
        folder_dataset = os.path.join("examples", "datasets")

    strat = 'balanced' if stratify else 'random'
    if args.dataset == 'femnist' and -1 in points_per_party:
        strat = 'orig_dist'

    if create_new:
        folder_party_data = os.path.join(folder_party_data, exp_name if exp_name else str(
            int(time.time())) + '_' + strat)
    else:
        folder_party_data = os.path.join(folder_party_data, dataset, strat)

    if not os.path.exists(folder_party_data):
        os.makedirs(folder_party_data)
    else:
        # clear folder of old data
        for f_name in os.listdir(folder_party_data):
            f_path = os.path.join(folder_party_data, f_name)
            if os.path.isfile(f_path):
                os.unlink(f_path)

    # Save new files
    if dataset == 'nursery':
        save_nursery_party_data(
            points_per_party, stratify, folder_party_data, folder_dataset)
    elif dataset == 'adult':
        save_adult_party_data(points_per_party, stratify,
                              folder_party_data, folder_dataset)
    elif dataset == 'german':
        save_german_party_data(points_per_party, stratify,
                               folder_party_data, folder_dataset)
    elif args.dataset == 'mnist':
        save_mnist_party_data(points_per_party, stratify,
                              folder_party_data, folder_dataset)
    elif args.dataset == 'compas':
        save_compas_party_data(points_per_party, stratify,
                               folder_party_data, folder_dataset)
    elif dataset == 'higgs':
        save_higgs_party_data(points_per_party, stratify,
                              folder_party_data, folder_dataset)
    elif dataset == 'airline':
        save_airline_party_data(
            points_per_party, stratify, folder_party_data, folder_dataset)
    elif dataset == 'diabetes':
        save_diabetes_party_data(
            points_per_party, stratify, folder_party_data, folder_dataset)
    elif dataset == 'binovf':
        save_binovf_party_data(points_per_party, stratify,
                               folder_party_data, folder_dataset)
    elif dataset == 'multovf':
        save_multovf_party_data(
            points_per_party, stratify, folder_party_data, folder_dataset)
    elif dataset == 'linovf':
        save_linovf_party_data(
            points_per_party, folder_party_data, folder_dataset)
    elif dataset == 'federated-clustering':
        save_federated_clustering_data(points_per_party, folder_party_data)
    elif dataset == 'femnist':
        save_femnist_party_data(
            points_per_party, stratify, folder_party_data, folder_dataset)
    elif dataset == 'cifar10':
        save_cifar10_party_data(
            points_per_party, stratify, folder_party_data, folder_dataset)
