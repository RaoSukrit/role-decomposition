from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
from random import shuffle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import sys
import os
import json
import time
import math

import pickle

import argparse

from sklearn import metrics

from tasks import *
from training import *
from models import *
from evaluation import *
from role_assignment_functions import *
from rolelearner.role_learning_tensor_product_encoder import RoleLearningTensorProductEncoder

import numpy as np

# Code for performing a tensor product decomposition on an
# existing set of vectors

use_cuda = torch.cuda.is_available()

if use_cuda:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--data_prefix", help="prefix for the vectors", type=str, default=None)
parser.add_argument("--role_prefix", help="prefix for a file of roles (if used)", type=str,
                    default=None)
parser.add_argument("--role_scheme", help="pre-coded role scheme to use", type=str, default=None)
parser.add_argument("--test_decoder", help="whether to test the decoder (in addition to MSE",
                    type=str, default="False")
parser.add_argument("--decoder", help="decoder type", type=str, default="ltr")
parser.add_argument("--decoder_prefix", help="prefix for the decoder to test", type=str,
                    default=None)
parser.add_argument("--decoder_embedding_size", help="embedding size for decoder", type=int,
                    default=20)
parser.add_argument("--decoder_task", help="task performed by the decoder", type=str,
                    default="auto")
parser.add_argument("--filler_dim", help="embedding dimension for fillers", type=int, default=10)
parser.add_argument("--role_dim", help="embedding dimension for roles", type=int, default=6)
parser.add_argument("--vocab_size", help="vocab size for the training language", type=int,
                    default=10)
parser.add_argument("--hidden_size", help="size of the encodings", type=int, default=60)
parser.add_argument("--save_vectors",
                    help="whether to save vectors generated by the fitted TPR model", type=str,
                    default="False")
parser.add_argument("--save_role_dicts",
                    help="whether to save role_to_index and index_to_role or not", type=str,
                    default="False")
parser.add_argument("--embedding_file", help="file containing pretrained embeddings", type=str,
                    default=None)
parser.add_argument("--unseen_words",
                    help="if using pretrained embeddings: whether to use all zeroes for unseen "
						 "words' embeddings, or to give them random vectors",
                    type=str, default="zero")
parser.add_argument("--extra_test_set", help="additional file to print predictions for", type=str,
                    default=None)
parser.add_argument("--train", help="whether or not to train the model", type=str, default="True")
parser.add_argument("--neighbor_analysis", help="whether to use a neighbor analysis", type=str,
                    default="True")
parser.add_argument("--digits", help="whether this is one of the digit task", type=str,
                    default="True")
parser.add_argument("--final_linear", help="whether to have a final linear layer", type=str,
                    default="True")
parser.add_argument("--embed_squeeze", help="original dimension to be squeezed to filler_dim",
                    type=int, default=None)
parser.add_argument("--role_learning", help="A flag for whether to enable role learning or use "
                                            "the provided roles.", action="store_true")
parser.add_argument("--bidirectional", help="A flag for whether the role learning module is "
                                            "bidirectional.", action="store_true")
parser.add_argument(
    "--role_assignment_shrink_filler_dim",
    help="If specified, the filler embedding is shrunk to this size before being input to the "
         "role assignment LSTM.",
    default=None,
    type=int)
parser.add_argument(
    "--use_one_hot_temperature",
    help="A flag for whether role learning one hot regularization should have an increasing "
         "temperature.",
    action="store_true")
parser.add_argument(
    "--role_assigner_num_layers",
    help="The number of layers for the role assignment network.",
    default=1,
    type=int
)
parser.add_argument(
    "--output_dir",
    help="An optional output folder where files can be saved to.",
    type=str,
    default=None
)
parser.add_argument(
    "--patience",
    help="The number of epochs to train if validation loss isn't improving",
    type=int,
    default=10
)
parser.add_argument(
    "--pretrained_filler_embedding",
    help="A weight file containing a pretrained filler embedding",
    type=str,
    default=None
)
parser.add_argument(
    "--softmax_roles",
    help="Whether the role predictions should be run through a softmax",
    action="store_true"
)
parser.add_argument(
    "--batch_size",
    help="The batch size.",
    default=32,
    type=int
)
parser.add_argument(
    "--one_hot_regularization_weight",
    help="The weight applied to the one hot regularization term",
    type=float,
    default=1.0
)
parser.add_argument(
    "--l2_norm_regularization_weight",
    help="The weight applied to the l2 norm regularization term",
    type=float,
    default=1.0
)
parser.add_argument(
    "--unique_role_regularization_weight",
    help="The weight applied to the unique role regularization term",
    type=float,
    default=1.0
)
parser.add_argument(
    "--num_roles",
    help="The number of roles to give a role learning network. This value is only used when "
         "--role_learning is enabled.",
    type=int,
    default=None
)
parser.add_argument(
    "--data_path",
    help="The location of the data files.",
    type=str,
    default="data"
)
parser.add_argument(
    "--burn_in",
    help="The number of epochs to train without regularization",
    type=int,
    default=0
)
parser.add_argument(
    "--shuffle",
    help="Whether to shuffle the input fillers and corresponding embedding. This is to generate a"
         " value by which to normalize the MSE to compare different role schemes across "
         "different tasks.",
    action="store_true"
)
parser.add_argument(
    "--scan_checkpoint",
    help="The location of a SCAN checkpoint. Settings this argument enables the SCAN task.",
    type=str,
    default=None,
    required=False
)
parser.add_argument(
    "--cogs_src_vocab",
    help="The location of the src vocab used while training the COGS model",
    type=str,
    default=None,
    required=False
)
parser.add_argument(
    "--cogs_tgt_vocab",
    help="The location of the tgt vocab used while training the COGS model",
    type=str,
    default=None,
    required=False
)


args = parser.parse_args()

output_dir = None
if args.output_dir:
    output_dir = os.path.join('output/', args.output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'arguments.txt'), 'w') as arguments_file:
        for argument, value in sorted(vars(args).items()):
            arguments_file.write('{}: {}\n'.format(argument, value))

print("**** Finished Creating Output Dir ****")

# Create the logfile
if output_dir:
    results_page = open(os.path.join(output_dir, 'log.txt'), 'w')
else:
    if args.final_linear != "True":
        results_page = open(
            "logs/" + args.data_prefix.split("/")[-1] + str(args.role_prefix).split("/")[-1] + str(
                args.role_scheme) + ".filler" + str(args.filler_dim) + ".role" + str(
                args.role_dim) + ".tpr_decomp.nf", "w")
    else:
        results_page = open(
            "logs/" + args.data_prefix.split("/")[-1] + str(args.role_prefix).split("/")[-1] + str(
                args.role_scheme) + ".filler" + str(args.filler_dim) + ".role" + str(
                args.role_dim) + "." + str(args.embed_squeeze) + ".tpr_decomp", "w")

print("**** Finished Creating LOG Dir ****")

# Load the decoder for computing swapping accuracy
if args.test_decoder == "True" and not args.scan_checkpoint:
    if args.decoder == "ltr":
        decoder = DecoderRNN(args.vocab_size, args.decoder_embedding_size, args.hidden_size)
    elif args.decoder == "bi":
        decoder = DecoderBiRNN(args.vocab_size, args.decoder_embedding_size, args.hidden_size)
    elif args.decoder == "tree":
        decoder = DecoderTreeRNN(args.vocab_size, args.decoder_embedding_size, args.hidden_size)
    else:
        print("Invalid decoder type")

    input_to_output = lambda seq: transform(seq, args.decoder_task)

    if use_cuda:
        decoder.load_state_dict(torch.load("models/decoder_" + args.decoder_prefix + ".weights"))
    else:
        decoder.load_state_dict(
            torch.load("models/decoder_" + args.decoder_prefix + ".weights", map_location='cpu')
        )

    if use_cuda:
        decoder = decoder.cuda()
elif args.test_decoder == "True" and args.scan_checkpoint:
    decoder = SCANDecoderRNN(
        hidden_size=args.hidden_size,
        output_size=args.vocab_size,
        mytype='GRU',
        n_layers=1,
        dropout_p=0
    )
    decoder.eval()

    print('Loading checkpoint: ' + args.scan_checkpoint)
    if use_cuda:
        checkpoint = torch.load(args.scan_checkpoint)
    else:
        checkpoint = torch.load(args.scan_checkpoint, map_location=lambda storage, loc: storage)

    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    if use_cuda:
        decoder = decoder.cuda()

    pairs_test = checkpoint['pairs_test']
    input_to_output_dict = {}
    for input_, output in pairs_test:
        input_to_output_dict[input_] = output

    input_to_output = lambda key: input_to_output_dict[key]

    input_lang = checkpoint['input_lang']
    output_lang = checkpoint['output_lang']

print("**** Finished Loading Decoder ****")

# Prepare the train, dev, and test data
unindexed_train = []
unindexed_dev = []
unindexed_test = []
unindexed_extra = []

filler_to_index = {}
index_to_filler = {}
role_to_index = {}
index_to_role = {}

filler_counter = 0
role_counter = 0
max_length = 0

if args.cogs_src_vocab:
    with open(args.cogs_src_vocab, 'r') as json_fh:
        src_filler_to_index = json.load(json_fh)

    src_filler_counter = len(src_filler_to_index)
    src_index_to_filler = {idx: token for token, idx in src_filler_to_index.items()}

    filler_counter = src_filler_counter

if args.cogs_tgt_vocab:
    with open(args.cogs_tgt_vocab, 'r') as json_fh:
        tgt_filler_to_index = json.load(json_fh)

    tgt_filler_counter = len(tgt_filler_to_index)
    tgt_index_to_filler = {idx: token for token, idx in tgt_filler_to_index.items()}

    # filler_counter += tgt_filler_counter

train_file = open(os.path.join(args.data_path, args.data_prefix + ".data_from_train"), "r")
for line in train_file:
    sequence, vector = line.strip().split("\t")
    unindexed_train.append(([value for value in sequence.split()], Variable(
        torch.FloatTensor(np.array([float(value) for value in vector.split()])))))

    if len(sequence.split()) > max_length:
        max_length = len(sequence.split())

    if not args.cogs_src_vocab:
        for filler in sequence.split():
            if filler not in filler_to_index:
                filler_to_index[filler] = filler_counter
                index_to_filler[filler_counter] = filler
                filler_counter += 1


print("**** Finished Loading Train Dataset ****")

dev_file = open(os.path.join(args.data_path, args.data_prefix + ".data_from_dev"), "r")
for line in dev_file:
    sequence, vector = line.strip().split("\t")
    unindexed_dev.append(([value for value in sequence.split()], Variable(
        torch.FloatTensor(np.array([float(value) for value in vector.split()])))))

    if len(sequence.split()) > max_length:
        max_length = len(sequence.split())

    if not args.cogs_src_vocab:
        for filler in sequence.split():
            if filler not in filler_to_index:
                filler_to_index[filler] = filler_counter
                index_to_filler[filler_counter] = filler
                filler_counter += 1

print("**** Finished Loading Dev Dataset ****")

if args.shuffle:
    print("Shuffling the input sequences and corresponding embeddings")
    sequences = []
    embeddings = []
    for sequence, embedding in unindexed_train:
        sequences.append(sequence)
        embeddings.append(embedding)
    shuffle(sequences)
    unindexed_train = []
    for i in range(len(sequences)):
        unindexed_train.append((sequences[i], embeddings[i]))

    sequences = []
    embeddings = []
    for sequence, embedding in unindexed_dev:
        sequences.append(sequence)
        embeddings.append(embedding)
    shuffle(sequences)
    unindexed_dev = []
    for i in range(len(sequences)):
        unindexed_dev.append((sequences[i], embeddings[i]))

test_file = open(os.path.join(args.data_path, args.data_prefix + ".data_from_test"), "r")
for line in test_file:
    sequence, vector = line.strip().split("\t")
    unindexed_test.append(([value for value in sequence.split()], Variable(
        torch.FloatTensor(np.array([float(value) for value in vector.split()])))))

    if len(sequence.split()) > max_length:
        max_length = len(sequence.split())

    if not args.cogs_tgt_vocab:
        for filler in sequence.split():
            if filler not in filler_to_index:
                filler_to_index[filler] = filler_counter
                index_to_filler[filler_counter] = filler
                filler_counter += 1


print("**** Finished Loading Test Dataset ****")

if args.extra_test_set is not None:
    extra_file = open(os.path.join(args.data_path, args.extra_test_set), "r")
    for line in extra_file:
        sequence, vector = line.strip().split("\t")
        unindexed_extra.append(([value for value in sequence.split()], Variable(
            torch.FloatTensor(np.array([float(value) for value in vector.split()]))).cuda()))

        if len(sequence.split()) > max_length:
            max_length = len(sequence.split())

        for filler in sequence.split():
            if filler not in filler_to_index:
                filler_to_index[filler] = filler_counter
                index_to_filler[filler_counter] = filler
                filler_counter += 1

if args.digits == "True":
    for i in range(10):
        filler_to_index[str(i)] = i
        index_to_filler[i] = str(i)

if not args.cogs_src_vocab:
    indexed_train = [([filler_to_index[filler] for filler in elt[0]], elt[1]) for elt in
                 unindexed_train]
    indexed_dev = [([filler_to_index[filler] for filler in elt[0]], elt[1]) for elt in unindexed_dev]
else:
    indexed_train = [([src_filler_to_index[filler] for filler in elt[0]], elt[1]) for elt in
                 unindexed_train]
    indexed_dev = [([src_filler_to_index[filler] for filler in elt[0]], elt[1]) for elt in unindexed_dev]

if not args.cogs_tgt_vocab:
    indexed_test = [([filler_to_index[filler] for filler in elt[0]], elt[1]) for elt in unindexed_test]
    indexed_extra = [([filler_to_index[filler] for filler in elt[0]], elt[1]) for elt in
                 unindexed_extra]
else:
    indexed_test = [([src_filler_to_index[filler] for filler in elt[0]], elt[1]) for elt in unindexed_test]
    indexed_extra = [([src_filler_to_index[filler] for filler in elt[0]], elt[1]) for elt in
                 unindexed_extra]

unindexed_train_roles = []
unindexed_dev_roles = []
unindexed_test_roles = []
unindexed_extra_roles = []

n_r = -1

# If there is a file of roles for the fillers, load those roles
if args.role_prefix is not None:
    train_role_file = open(
        os.path.join(args.data_path, args.role_prefix + ".data_from_train.roles"), "r")
    for line in train_role_file:
        unindexed_train_roles.append(line.strip().split())
        for role in line.strip().split():
            if role not in role_to_index:
                role_to_index[role] = role_counter
                index_to_role[role_counter] = role
                role_counter += 1

    dev_role_file = open(
        os.path.join(args.data_path, args.role_prefix + ".data_from_dev.roles"), "r")
    for line in dev_role_file:
        unindexed_dev_roles.append(line.strip().split())
        for role in line.strip().split():
            if role not in role_to_index:
                role_to_index[role] = role_counter
                index_to_role[role_counter] = role
                role_counter += 1

    test_role_file = open(
        os.path.join(args.data_path, args.role_prefix + ".data_from_test.roles"), "r")
    for line in test_role_file:
        unindexed_test_roles.append(line.strip().split())
        for role in line.strip().split():
            if role not in role_to_index:
                role_to_index[role] = role_counter
                index_to_role[role_counter] = role
                role_counter += 1

    if args.extra_test_set is not None:
        extra_role_file = open(os.path.join(args.data_path, args.extra_test_set + ".roles"), "r")
        for line in extra_role_file:
            unindexed_extra_roles.append(line.strip().split())
            for role in line.strip().split():
                if role not in role_to_index:
                    role_to_index[role] = role_counter
                    index_to_role[role_counter] = role
                    role_counter += 1



# Or, if a predefined role scheme is being used, prepare it
elif args.role_scheme is not None:
    if args.role_scheme == "bow":
        n_r, seq_to_roles = create_bow_roles(max_length, len(filler_to_index.keys()))
    elif args.role_scheme == "ltr":
        n_r, seq_to_roles = create_ltr_roles(max_length, len(filler_to_index.keys()))
    elif args.role_scheme == "rtl":
        n_r, seq_to_roles = create_rtl_roles(max_length, len(filler_to_index.keys()))
    elif args.role_scheme == "bi":
        n_r, seq_to_roles = create_bidirectional_roles(max_length, len(filler_to_index.keys()))
    elif args.role_scheme == "wickel":
        n_r, seq_to_roles = create_wickel_roles(max_length, len(filler_to_index.keys()))
    elif args.role_scheme == "tree":
        n_r, seq_to_roles = create_tree_roles(max_length, len(filler_to_index.keys()))
    elif args.role_scheme == "interleave":
        n_r, seq_to_roles = create_interleaving_tree_roles(max_length, len(filler_to_index.keys()))
    else:
        print("Invalid role scheme")

    for pair in indexed_train:
        these_roles = seq_to_roles(pair[0])
        unindexed_train_roles.append(these_roles)
        for role in these_roles:
            if role not in role_to_index:
                role_to_index[role] = role
                index_to_role[role] = role
                role_counter += 1

    for pair in indexed_dev:
        these_roles = seq_to_roles(pair[0])
        unindexed_dev_roles.append(these_roles)
        for role in these_roles:
            if role not in role_to_index:
                role_to_index[role] = role
                index_to_role[role] = role
                role_counter += 1

    for pair in indexed_test:
        these_roles = seq_to_roles(pair[0])
        unindexed_test_roles.append(these_roles)
        for role in these_roles:
            if role not in role_to_index:
                role_to_index[role] = role
                index_to_role[role] = role
                role_counter += 1

    for pair in indexed_extra:
        these_roles = seq_to_roles(pair[0])
        unindexed_extra_roles.append(these_roles)
        for role in these_roles:
            if role not in role_to_index:
                role_to_index[role] = role
                index_to_role[role] = role
                role_counter += 1
else:
    print("No role scheme specified")

print("**** Finished Loading Pretrained ROLE Embeddings and ROLE Scheme ****")

# print("filler_to_index")
# import json
# with open('filler_to_index.json', 'w') as fh:
#     json.dump(filler_to_index, fh)
# # print(filler_to_index)
# # print(role_to_index)
# with open('role_to_index.json', 'w') as fh:
#     json.dump(role_to_index, fh)

indexed_train_roles = [[role_to_index[role] for role in roles] for roles in unindexed_train_roles]
indexed_dev_roles = [[role_to_index[role] for role in roles] for roles in unindexed_dev_roles]
indexed_test_roles = [[role_to_index[role] for role in roles] for roles in unindexed_test_roles]
indexed_extra_roles = [[role_to_index[role] for role in roles] for roles in unindexed_extra_roles]

all_train_data = []
all_dev_data = []
all_test_data = []
all_extra_data = []

if not args.role_learning:
    # Make sure the number of fillers and the number of roles always matches
    for index, element in enumerate(indexed_train):
        if len(element[0]) != len(indexed_train_roles[index]):
            print(index, "ERROR!!!", element[0], indexed_train_roles[index])
        else:
            all_train_data.append((element[0], indexed_train_roles[index], element[1]))

    for index, element in enumerate(indexed_dev):
        if len(element[0]) != len(indexed_dev_roles[index]):
            print(index, "ERROR!!!", element[0], indexed_dev_roles[index])
        else:
            all_dev_data.append((element[0], indexed_dev_roles[index], element[1]))

    for index, element in enumerate(indexed_test):
        if len(element[0]) != len(indexed_test_roles[index]):
            print(index, "ERROR!!!", element[0], indexed_test_roles[index])
        else:
            all_test_data.append((element[0], indexed_test_roles[index], element[1]))

    for index, element in enumerate(indexed_extra):
        if len(element[0]) != len(indexed_extra_roles[index]):
            print(index, "ERROR!!!", element[0], indexed_extra_roles[index])
        else:
            all_extra_data.append((element[0], indexed_extra_roles[index], element[1]))
else:
    # Add dummy roles to the training data, these will be ignored since we are role learning
    for index, element in enumerate(indexed_train):
        all_train_data.append((element[0], 0, element[1]))

    for index, element in enumerate(indexed_dev):
        all_dev_data.append((element[0], 0, element[1]))

    for index, element in enumerate(indexed_test):
        all_test_data.append((element[0], 0, element[1]))

    for index, element in enumerate(indexed_extra):
        all_extra_data.append((element[0], 0, element[1]))

weights_matrix = None

# Prepare the embeddings
# If a file of embeddings was provided, use those.
embedding_dict = None
if args.embedding_file is not None:
    embedding_dict = {}
    embed_file = open(args.embedding_file, "r")
    for line in embed_file:
        parts = line.strip().split()
        if len(parts) == args.filler_dim + 1:
            embedding_dict[parts[0]] = list(map(lambda x: float(x), parts[1:]))

    matrix_len = len(filler_to_index.keys())
    if args.embed_squeeze is not None:
        weights_matrix = np.zeros((matrix_len, args.embed_squeeze))
    else:
        weights_matrix = np.zeros((matrix_len, args.filler_dim))

    for i in range(matrix_len):
        word = index_to_filler[i]
        if word in embedding_dict:
            weights_matrix[i] = embedding_dict[word]
        else:
            if args.unseen_words == "random":
                weights_matrix[i] = np.random.normal(scale=0.6, size=(args.filler_dim,))
            elif args.unseen_words == "zero":
                pass  # It was initialized as zero, so don't need to do anything
            else:
                print("Invalid choice for embeddings of unseen words")

# Initialize the TPDN
if n_r != -1:
    role_counter = n_r

if args.final_linear == "True":
    final_layer_width = args.hidden_size
else:
    final_layer_width = None

if args.role_learning:
    if args.num_roles:
        role_counter = args.num_roles

    print(f"args.bidirectional={args.bidirectional}")
    print("Using RoleLearningTensorProductEncoder with {} roles".format(role_counter))
    tpr_encoder = RoleLearningTensorProductEncoder(
        n_roles=role_counter,
        n_fillers=filler_counter,
        final_layer_width=final_layer_width,
        filler_dim=args.filler_dim,
        role_dim=args.role_dim,
        pretrained_filler_embeddings=args.pretrained_filler_embedding,
        embedder_squeeze=args.embed_squeeze,
        role_assignment_shrink_filler_dim=args.role_assignment_shrink_filler_dim,
        bidirectional=args.bidirectional,
        num_layers=args.role_assigner_num_layers,
        softmax_roles=args.softmax_roles,
        pretrained_embeddings=weights_matrix,
        one_hot_regularization_weight=args.one_hot_regularization_weight,
        l2_norm_regularization_weight=args.l2_norm_regularization_weight,
        unique_role_regularization_weight=args.unique_role_regularization_weight,
    )
else:
    print("Using TensorProductEncoder")
    tpr_encoder = TensorProductEncoder(
        n_roles=role_counter,
        n_fillers=filler_counter,
        final_layer_width=final_layer_width,
        filler_dim=args.filler_dim,
        role_dim=args.role_dim,
        pretrained_embeddings=weights_matrix,
        embedder_squeeze=args.embed_squeeze,
        pretrained_filler_embeddings=args.pretrained_filler_embedding
    )

if use_cuda:
    tpr_encoder = tpr_encoder.cuda()

print("**** Finished Loading Encoder ****")

#args.data_prefix = args.data_prefix.split("/")[-1] + ".filler" + str(
#    args.filler_dim) + ".role" + str(args.role_dim)
#if args.final_linear != "True":
#    args.data_prefix += ".no_final"

# Train the TPDN
args.role_prefix = str(args.role_prefix).split("/")[-1]
if args.train == "True":
    if output_dir:
        weight_file = os.path.join(output_dir, 'model.tpr')
    else:
        weight_file = "models/" + args.data_prefix + str(
                                  args.role_prefix) + str(args.role_scheme) + ".tpr"
    end_loss = trainIters_tpr(
        all_train_data,
        all_dev_data,
        tpr_encoder,
        n_epochs=1000,
        learning_rate=0.001,
        weight_file=weight_file,
        batch_size=args.batch_size,
        use_one_hot_temperature=args.use_one_hot_temperature,
        patience=args.patience,
        burn_in=args.burn_in
    )
print("**** Finished Training ****")

# Load the trained TPDN
weight_file = "/Users/sukritrao/Documents/NYU/Coursework/Summer2022/Research/role-analysis/role-decomposition/output/role_output_lstm_bi_1_embd/model.tpr"
tpr_encoder.load_state_dict(torch.load(weight_file, map_location=device))

# Prepare test data
all_test_data_orig = all_test_data
all_test_data = batchify_tpr(all_test_data, 1)

test_data_sets = [(Variable(torch.LongTensor([item[0] for item in batch])),
                   Variable(torch.LongTensor([item[1] for item in batch])),
                   torch.cat([item[2].unsqueeze(0).unsqueeze(0) for item in batch], 1)) for
                  batch in all_test_data]

neighbor_counter = 0
neighbor_total_rank = 0
neighbor_correct = 0

# Evaluate on test set
test_symbolic_mse = 0
test_continuous_mse = 0
test_one_hot_loss = 0
test_l2_loss = 0
test_unique_role_loss = 0

for i in range(len(test_data_sets)):
    input_fillers = test_data_sets[i][0]
    input_roles = test_data_sets[i][1]
    target_variable = test_data_sets[i][2]
    if use_cuda:
        input_fillers = input_fillers.cuda()
        input_roles = input_roles.cuda()
        target_variable = target_variable.cuda()
    if isinstance(tpr_encoder, RoleLearningTensorProductEncoder):
        tpr_encoder.eval()
        encoding, role_predictions = tpr_encoder(input_fillers, input_roles)
        test_symbolic_mse += torch.mean(torch.pow(encoding.data - target_variable.data, 2))
        tpr_encoder.train()
        encoding, role_predictions = tpr_encoder(input_fillers, input_roles)
        test_continuous_mse += torch.mean(torch.pow(encoding.data - target_variable.data, 2))

        batch_one_hot_loss, batch_l2_norm_loss, batch_unique_role_loss = \
            tpr_encoder.get_regularization_loss(role_predictions)
        test_one_hot_loss += batch_one_hot_loss
        test_l2_loss += batch_l2_norm_loss
        test_unique_role_loss += batch_unique_role_loss
    else:
        encoding = tpr_encoder(input_fillers, input_roles)
        test_symbolic_mse += torch.mean(torch.pow(encoding.data - target_variable.data, 2))

test_symbolic_mse = test_symbolic_mse / len(all_test_data)
test_continuous_mse = test_continuous_mse / len(all_test_data)
test_one_hot_loss = test_one_hot_loss / len(all_test_data)
test_l2_loss = test_l2_loss / len(all_test_data)
test_unique_role_loss = test_unique_role_loss / len(all_test_data)

total_symbolic_test_loss = \
    test_symbolic_mse + test_one_hot_loss + test_l2_loss + test_unique_role_loss
total_continuous_test_loss = \
    test_continuous_mse + test_one_hot_loss + test_l2_loss + test_unique_role_loss

print("**** Finished Computing Test Metrics ****")

if args.output_dir:
    results_page.write(output_dir + "\n")

if args.test_decoder == "True" and not args.scan_checkpoint:
    if isinstance(tpr_encoder, RoleLearningTensorProductEncoder):
        tpr_encoder.eval()
    correct, total = score2(tpr_encoder, decoder, input_to_output, batchify(all_test_data, 1),
                            index_to_filler)
    results_page.write(args.data_prefix + str(args.role_prefix) + str(args.role_scheme) +
                       ".tpr" + " Discrete Swapping encoder performance: " + str(correct)
                       + " " + str(total) + "\n")
    if isinstance(tpr_encoder, RoleLearningTensorProductEncoder):
        tpr_encoder.train()
        correct, total = score2(tpr_encoder, decoder, input_to_output, batchify(all_test_data, 1),
                                index_to_filler)
        results_page.write(args.data_prefix + str(args.role_prefix) + str(args.role_scheme) +
                           ".tpr" + " Continuous Swapping encoder performance: " + str(correct)
                           + " " + str(total) + "\n")

elif args.test_decoder == "True" and args.scan_checkpoint:
    if isinstance(tpr_encoder, RoleLearningTensorProductEncoder):
        tpr_encoder.eval()
    correct, total = scoreSCAN(tpr_encoder, decoder, input_to_output, batchify(all_test_data, 1),
                               index_to_filler, output_lang)
    results_page.write('Discrete swapping performance: {} {}\n'.format(correct, total))
    if isinstance(tpr_encoder, RoleLearningTensorProductEncoder):
        tpr_encoder.train()
        correct, total = scoreSCAN(tpr_encoder, decoder, input_to_output, batchify(all_test_data, 1),
                                   index_to_filler, output_lang)
        results_page.write('Continuous swapping performance: {} {}\n'.format(correct, total))

results_page.write('Test symbolic MSE loss: {}\n'.format(test_symbolic_mse))
results_page.write('Test symbolic loss: {}\n'.format(total_symbolic_test_loss.item()))
if isinstance(tpr_encoder, RoleLearningTensorProductEncoder):
    results_page.write('Test continuous MSE loss: {}\n'.format(test_continuous_mse))
    results_page.write('Test continuous loss: {}\n'.format(total_continuous_test_loss.item()))
results_page.write('Test one hot loss: {}\n'.format(test_one_hot_loss))
results_page.write('Test unique role loss: {}\n'.format(test_unique_role_loss))
results_page.write('Test l2 norm loss: {}\n'.format(test_l2_loss))

# Run cluster performance analysis
if isinstance(tpr_encoder, RoleLearningTensorProductEncoder):
    tpr_encoder.train()
    role_assigner = tpr_encoder.role_assigner
    roles_true = []
    roles_predicted = []
    role_indices = torch.tensor([x for x in range(role_counter)], device=device)
    role_embeddings = role_assigner.role_embedding(role_indices)
    num_elements = 0
    num_elements_role_low = 0
    for test_example in all_test_data:

        sequence = test_example[0][0]
        roles_true.extend(test_example[0][1])
        role_emb, role_weights = role_assigner(torch.tensor([sequence], device=device))
        for i in range(len(role_weights)):
            role_prediction = np.argmax(role_weights[i][0].cpu().detach().numpy())
            if role_weights[i][0][role_prediction] < .98:
                num_elements_role_low += 1
            num_elements += 1
            roles_predicted.append(role_prediction)

        # Get the predicted role using cosine distance from the role embeddings
        #for i in range(len(role_emb)):
        #    similarities = []
        #    for j in range(role_counter):
        #        similarities.append(F.cosine_similarity(role_emb[i][0], role_embeddings[j], dim=0))
        #    roles_predicted.append(np.argmax(similarities))

    results_page.write(
        'number of roles used: {}\n'.format(len(np.unique(roles_predicted)))
    )
    results_page.write(
        'percentage low role prediction: {}\n'.format(100 * num_elements_role_low / num_elements)
    )
    results_page.write(
        'adjusted rand index: {}\n'.format(metrics.adjusted_rand_score(roles_true, roles_predicted))
    )
    results_page.write(
        'normalized mutual info: {}\n'.format(metrics.normalized_mutual_info_score(roles_true, roles_predicted))
    )
    results_page.write(
        'adjusted mutual info: {}\n'.format(metrics.adjusted_mutual_info_score(roles_true, roles_predicted))
    )
    results_page.write(
        'homogenity: {}\n'.format(metrics.homogeneity_score(roles_true, roles_predicted))
    )
    results_page.write(
        'completeness: {}\n'.format(metrics.completeness_score(roles_true, roles_predicted))
    )
    results_page.write(
        'v measure score: {}\n'.format(metrics.v_measure_score(roles_true, roles_predicted))
    )
    results_page.write(
        'fowlkes mallows score: {}\n'.format(metrics.fowlkes_mallows_score(roles_true, roles_predicted))
    )
    results_page.write('\n')

# Generate role predictions
if isinstance(tpr_encoder, RoleLearningTensorProductEncoder):
    print('Generating role predictions')
    tpr_encoder.train()
    role_assigner = tpr_encoder.role_assigner
    output_dir = os.path.join('output/', args.output_dir)
    basename = os.path.basename(output_dir)
    train_data_file = open(os.path.join(args.data_path, args.data_prefix + '.data_from_train'), 'r')
    train_role_file = open(
        os.path.join(output_dir, args.data_prefix + '.' + basename + '.data_from_train.roles'), 'w')

    dev_data_file = open(os.path.join(args.data_path, args.data_prefix + '.data_from_dev'), 'r')
    dev_role_file = open(
        os.path.join(output_dir, args.data_prefix + '.' + basename + '.data_from_dev.roles'), 'w')

    test_data_file = open(os.path.join(args.data_path, args.data_prefix + '.data_from_test'), 'r')
    test_role_file = open(
        os.path.join(output_dir, args.data_prefix + '.' + basename + '.data_from_test.roles'), 'w')

    for line in train_data_file:
        sequence, embedding = line.strip().split('\t')
        sequence = sequence.split()
        sequence = list(map(lambda filler: filler_to_index[filler], sequence))
        sequence = torch.LongTensor([sequence])
        if use_cuda:
            sequence = sequence.cuda()
        role_emb, role_weights = role_assigner(sequence)
        roles = []
        for i in range(len(role_weights)):
            roles.append(str(np.argmax(role_weights[i].cpu().detach().numpy())))
        train_role_file.write(' '.join(roles) + '\n')
    train_role_file.close()

    for line in dev_data_file:
        sequence, embedding = line.strip().split("\t")
        sequence = sequence.split()
        sequence = list(map(lambda filler: filler_to_index[filler], sequence))
        sequence = torch.LongTensor([sequence])
        if use_cuda:
            sequence = sequence.cuda()
        role_emb, role_weights = role_assigner(sequence)
        roles = []
        for i in range(len(role_weights)):
            roles.append(str(np.argmax(role_weights[i].cpu().detach().numpy())))
        dev_role_file.write(' '.join(roles) + '\n')
    dev_role_file.close()

    for line in test_data_file:
        sequence, embedding = line.strip().split("\t")
        sequence = sequence.split()
        sequence = list(map(lambda filler: filler_to_index[filler], sequence))
        sequence = torch.LongTensor([sequence])
        if use_cuda:
            sequence = sequence.cuda()
        role_emb, role_weights = role_assigner(sequence)
        roles = []
        for i in range(len(role_weights)):
            roles.append(str(np.argmax(role_weights[i].cpu().detach().numpy())))
        test_role_file.write(' '.join(roles) + '\n')
    test_role_file.close()

# Save the test set predictions, if desired
if args.save_vectors == "True":
    fo_pred = open(
        args.data_prefix + str(args.role_prefix) + str(args.role_scheme) + ".tpr.test_preds", "w")

    for i in range(len(test_data_sets)):
        sequence = all_test_data[i][0]
        pred = tpr_encoder(test_data_sets[i][0], test_data_sets[i][1]).data.cpu().numpy()[0][0]

        sequence = [str(x) for x in sequence]
        pred = [str(x) for x in pred]

        fo_pred.write(" ".join(sequence) + "\t" + " ".join(pred) + "\n")

# Save the role dictionaries, if desired
if args.save_role_dicts == "True":
    with open(args.data_prefix + str(args.role_prefix) + str(
            args.role_scheme) + '.role_to_index.pickle', 'wb') as handle:
        pickle.dump(role_to_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(args.data_prefix + str(args.role_prefix) + str(
            args.role_scheme) + '.index_to_role.pickle', 'wb') as handle:
        pickle.dump(index_to_role, handle, protocol=pickle.HIGHEST_PROTOCOL)
