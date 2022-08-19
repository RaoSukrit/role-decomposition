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

import time
import math

import collections
import json

import pickle

from binding_operations import *
from role_assignment_functions import *

# Definitions of all the seq2seq models and the TPDN

use_cuda = torch.cuda.is_available()

# Encoder RNN for the mystery vector generating network--unidirectional GRU
class EncoderRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size # Hidden size
        self.embedding = nn.Embedding(input_size, emb_size) # Embedding layer
        self.rnn = nn.GRU(emb_size, hidden_size) # Recurrent layer

    # A forward pass of the encoder
    def forward(self, sequence):
        hidden = self.init_hidden(len(sequence))
        batch_size = len(sequence)
        sequence = Variable(torch.LongTensor([sequence])).transpose(0,2)#.cuda()

        if use_cuda:
            sequence = sequence.cuda()

        for element in sequence:
            if use_cuda:
                embedded = self.embedding(element).transpose(0,1)
            else:
                embedded = self.embedding(element).transpose(0,1)
            output, hidden = self.rnn(embedded, hidden)

        return hidden

    # Initialize the hidden state as all zeroes
    def init_hidden(self, batch_size):
        result = Variable(torch.zeros(1,batch_size,self.hidden_size))

        if use_cuda:
            return result.cuda()
        else:
            return result

# Encoder RNN for the mystery vector generating network--bidirectional GRU
class EncoderBiRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size):
        super(EncoderBiRNN, self).__init__()
        self.hidden_size = hidden_size # Hidden size
        self.embedding = nn.Embedding(input_size, emb_size) # Embedding layer
        self.rnn_fwd = nn.GRU(emb_size, int(hidden_size/2)) # Recurrent layer-forward
        self.rnn_rev = nn.GRU(emb_size, int(hidden_size/2)) # Recurrent layer-backward

    # A forward pass of the encoder
    def forward(self, sequence):
        batch_size = len(sequence)

        sequence_rev = Variable(torch.LongTensor([sequence[::-1]])).transpose(0,2)
        if use_cuda:
            sequence_rev = sequence_rev.cuda()


        sequence = Variable(torch.LongTensor([sequence])).transpose(0,2)
        if use_cuda:
            sequence = sequence.cuda()

        # Forward pass
        hidden_fwd = self.init_hidden(batch_size)

        for element in sequence:
            embedded = self.embedding(element).transpose(0,1)
            output, hidden_fwd = self.rnn_fwd(embedded, hidden_fwd)

        # Backward pass
        hidden_rev = self.init_hidden(batch_size)

        for element in sequence_rev:
            embedded = self.embedding(element).transpose(0,1)
            output, hidden_rev = self.rnn_rev(embedded, hidden_rev)

        # Concatenate the two hidden representations
        hidden = torch.cat((hidden_fwd, hidden_rev), 2)

        return hidden

    # Initialize the hidden state as all zeroes
    def init_hidden(self, batch_size):
        result = Variable(torch.zeros(1,batch_size,int(self.hidden_size/2)))

        if use_cuda:
            return result.cuda()
        else:
            return result

# Encoder RNN for the mystery vector generating network--Tree-GRU.
# Based on Chen et al. (2017): Improved neural machine translation
# with a syntax-aware encoder and decoder.
class EncoderTreeRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        super(EncoderTreeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.embedding = nn.Embedding(vocab_size, emb_size)

        self.w_z = nn.Linear(emb_size, hidden_size)
        self.u_zl = nn.Linear(hidden_size, hidden_size)
        self.u_zr = nn.Linear(hidden_size, hidden_size)
        self.w_r = nn.Linear(emb_size, hidden_size)
        self.u_rl = nn.Linear(hidden_size, hidden_size)
        self.u_rr = nn.Linear(hidden_size, hidden_size)
        self.w_h = nn.Linear(emb_size, hidden_size)
        self.u_hl = nn.Linear(hidden_size, hidden_size)
        self.u_hr = nn.Linear(hidden_size, hidden_size)

    def tree_gru(self, word, hidden_left, hidden_right):
        z_t = nn.Sigmoid()(self.w_z(word) + self.u_zl(hidden_left) + self.u_zr(hidden_right))
        r_t = nn.Sigmoid()(self.w_r(word) + self.u_rl(hidden_left) + self.u_rr(hidden_right))
        h_tilde = F.tanh(self.w_h(word) + self.u_hl(r_t * hidden_left) + self.u_hr(r_t * hidden_right))
        h_t = z_t * hidden_left + z_t * hidden_right + (1 - z_t) * h_tilde

        return h_t

    def forward(self, input_batch):
        final_output = None
        for input_seq in input_batch:
            tree = parse_digits(input_seq)

            embedded_seq = []

            for elt in input_seq:
                embedded_seq.append(self.embedding(Variable(torch.LongTensor([elt])).cuda()).unsqueeze(0))

            leaf_nodes = []
            for elt in embedded_seq:
                this_hidden = self.tree_gru(elt, self.init_hidden(), self.init_hidden())
                leaf_nodes.append(this_hidden)

            current_level = leaf_nodes
            for level in tree:
                next_level = []

                for node in level:

                    if len(node) == 1:
                        next_level.append(current_level[node[0]])
                        continue
                    left = node[0]
                    right = node[1]

                    hidden = self.tree_gru(self.init_word(), current_level[left], current_level[right])

                    next_level.append(hidden)

                current_level = next_level
            if final_output is None:
                final_output = current_level[0][0].unsqueeze(0)
            else:
                final_output = torch.cat((final_output, current_level[0][0].unsqueeze(0)),0)

        return final_output.transpose(0,1)

    # Initialize the hidden state as all zeroes
    def init_hidden(self):
        result = Variable(torch.zeros(1,1,int(self.hidden_size)))

        if use_cuda:
            return result.cuda()
        else:
            return result


    # Initialize the word hidden state as all zeroes
    def init_word(self):
        result = Variable(torch.zeros(1,1,int(self.emb_size)))

        if use_cuda:
            return result.cuda()
        else:
            return result


# Bidirectional decoder RNN for the mystery vector decoding network
# At each step of decoding, the decoder takes the encoding of the
# input (i.e. the final hidden state of the encoder) as well as
# the previous hidden state. It outputs a probability distribution
# over the possible output digits; the highest-probability digit is
# taken to be that time step's output
class DecoderBiRNN(nn.Module):
    def __init__(self, output_size, emb_size, hidden_size):
        super(DecoderBiRNN, self).__init__()
        self.hidden_size = hidden_size # Size of the hidden state
        self.output_size = output_size # Size of the output
        self.emb_size = emb_size
        self.rnn_fwd = nn.GRU(emb_size, int(hidden_size/2)) # Recurrent layer-forward
        self.rnn_rev = nn.GRU(emb_size, int(hidden_size/2)) # Recurrent layer-backward
        self.out = nn.Linear(hidden_size, output_size) # Linear layer giving the output
        self.softmax = nn.LogSoftmax() # Softmax layer
        self.squeeze = nn.Linear(hidden_size, int(hidden_size/2))

    # Forward pass
    def forward(self, hidden, output_len, tree):
        outputs = []
        encoder_hidden = self.squeeze(F.relu(hidden))
        fwd_hiddens = []
        rev_hiddens = []

        fwd_hidden = encoder_hidden
        for item in range(output_len):
            if use_cuda:
                output, fwd_hidden = self.rnn_fwd(Variable(torch.zeros(1,fwd_hidden.size()[1],int(self.emb_size))).cuda(), fwd_hidden) # Pass the inputs through the hidden layer
            else:
                output, fwd_hidden = self.rnn_fwd(Variable(torch.zeros(1,fwd_hidden.size()[1],int(self.emb_size))), fwd_hidden)
            fwd_hiddens.append(fwd_hidden)

        rev_hidden = encoder_hidden
        for item in range(output_len):
            if use_cuda:
                output, rev_hidden = self.rnn_rev(Variable(torch.zeros(1,rev_hidden.size()[1],int(self.emb_size))).cuda(), rev_hidden) # Pass the inputs through the hidden layer
            else:
                output, rev_hidden = self.rnn_rev(Variable(torch.zeros(1,rev_hidden.size()[1],int(self.emb_size))), rev_hidden)
            rev_hiddens.append(rev_hidden)

        all_hiddens = zip(fwd_hiddens, rev_hiddens[::-1])

        for hidden_pair in all_hiddens:
            output = torch.cat((hidden_pair[0], hidden_pair[1]), 2)
            output = self.softmax(self.out(output[0])) # Pass the result through softmax to make it probabilities
            outputs.append(output)

        return outputs


# Tree-based seq2seq decoder.
# Based on Chen et al. (2018): Tree-to-tree neural networks for program translation.
class DecoderTreeRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        super(DecoderTreeRNN, self).__init__()
        self.hidden_size = hidden_size
        self.word_out = nn.Linear(hidden_size, vocab_size)
        self.left_child = nn.GRU(hidden_size, hidden_size)
        self.right_child = nn.GRU(hidden_size, hidden_size)

    def forward(self, encoding_list, output_len, tree_list):
        words_out = []
        for encoding_mini, tree in zip(encoding_list.transpose(0,1), tree_list):

            encoding = encoding_mini.unsqueeze(0)
            tree_to_use = tree[::-1][1:]

            current_layer = [encoding]

            for layer in tree_to_use:
                next_layer = []
                for index, node in enumerate(layer):
                    if len(node) == 1:
                        next_layer.append(current_layer[index])
                    else:
                        left_variable = Variable(torch.zeros(1,1,self.hidden_size))
                        right_variable = Variable(torch.zeros(1,1,self.hidden_size))
                        if use_cuda:
                            left_variable = left_variable.cuda()
                            right_variable = right_variable.cuda()
                        output, left = self.left_child(left_variable, current_layer[index])
                        output, right = self.right_child(right_variable, current_layer[index])
                        next_layer.append(left)
                        next_layer.append(right)
                current_layer = next_layer


            if words_out == []:
                for elt in current_layer:
                    words_out.append(nn.LogSoftmax()(self.word_out(elt).view(-1).unsqueeze(0)))
            else:
                index = 0
                for elt in current_layer:
                    words_out[index] = torch.cat((words_out[index], nn.LogSoftmax()(self.word_out(elt).view(-1).unsqueeze(0))), 0)
                    index += 1

        return words_out


# Unidirectional decoder RNN for the mystery vector decoding network
# At each step of decoding, the decoder takes the encoding of the
# input (i.e. the final hidden state of the encoder) as well as
# the previous hidden state. It outputs a probability distribution
# over the possible output digits; the highest-probability digit is
# taken to be that time step's output
# class DecoderRNN(nn.Module):
#     def __init__(self, output_size, emb_size, hidden_size):
#         super(DecoderRNN, self).__init__()
#         self.hidden_size = hidden_size # Size of the hidden state
#         self.output_size = output_size # Size of the output
#         self.emb_size = emb_size
#         self.rnn = nn.GRU(emb_size, hidden_size) # Recurrent unit
#         self.out = nn.Linear(hidden_size, output_size) # Linear layer giving the output
#         self.softmax = nn.LogSoftmax() # Softmax layer

#     # Forward pass
#     def forward(self, hidden, output_len, tree):
#         outputs = []
#         hidden = F.relu(hidden)

#         for item in range(output_len):
#             if use_cuda:
#                 output, hidden = self.rnn(Variable(torch.zeros(1,hidden.size()[1],int(self.emb_size))).cuda(), hidden) # Pass the inputs through the hidden layer
#             else:
#                 output, hidden = self.rnn(Variable(torch.zeros(1,hidden.size()[1],int(self.emb_size))), hidden)
#             output = self.softmax(self.out(output[0])) # Pass the result through softmax to make it probabilities
#             outputs.append(output)

#         return outputs

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# A tensor product encoder layer
# Takes a list of fillers and a list of roles and returns an encoding
class TensorProductEncoder(nn.Module):
    def __init__(self, n_roles=2, n_fillers=2, filler_dim=3, role_dim=4,
                 final_layer_width=None, pretrained_embeddings=None, embedder_squeeze=None,
                 binder="tpr", pretrained_filler_embeddings=None):

        super(TensorProductEncoder, self).__init__()

        self.n_roles = n_roles # number of roles
        self.n_fillers = n_fillers # number of fillers

        # Set the dimension for the filler embeddings
        self.filler_dim = filler_dim

        # Set the dimension for the role embeddings
        self.role_dim = role_dim

        # Create an embedding layer for the fillers
        if embedder_squeeze is None:
                self.filler_embedding = nn.Embedding(self.n_fillers, self.filler_dim)
                self.embed_squeeze = False
                print("no squeeze")
        else:
                self.embed_squeeze = True
                self.filler_embedding = nn.Embedding(self.n_fillers, embedder_squeeze)
                self.embedding_squeeze_layer = nn.Linear(embedder_squeeze, self.filler_dim)
                print("squeeze")

        if pretrained_embeddings is not None:
                self.filler_embedding.load_state_dict({'weight': torch.FloatTensor(pretrained_embeddings).cuda()})
                self.filler_embedding.weight.requires_grad = False

        if pretrained_filler_embeddings:
            print('Using pretrained filler embeddings')
            self.filler_embedding.load_state_dict(
                torch.load(pretrained_filler_embeddings, map_location=device)
            )
            self.filler_embedding.weight.requires_grad = False

        # Create an embedding layer for the roles
        self.role_embedding = nn.Embedding(self.n_roles, self.role_dim)

        # Create a SumFlattenedOuterProduct layer that will
        # take the sum flattened outer product of the filler
        # and role embeddings (or a different type of role-filler
        # binding function, such as circular convolution)
        if binder == "tpr":
            self.sum_layer = SumFlattenedOuterProduct()
        elif binder == "hrr":
            self.sum_layer = CircularConvolution(self.filler_dim)
        elif binder == "eltwise" or binder == "elt":
            self.sum_layer = EltWise()
        else:
            print("Invalid binder")

        # This final part if for including a final linear layer that compresses
        # the sum flattened outer product into the dimensionality you desire
        # But if self.final_layer_width is None, then no such layer is used
        self.final_layer_width = final_layer_width
        if self.final_layer_width is None:
            self.has_last = 0
        else:
            self.has_last = 1
            if binder == "tpr":
                self.last_layer = nn.Linear(self.filler_dim * self.role_dim, self.final_layer_width)
            else:
                self.last_layer = nn.Linear(self.filler_dim, self.final_layer_width)

    # Function for a forward pass through this layer. Takes a list of fillers and
    # a list of roles and returns an single vector encoding it.
    def forward(self, filler_list, role_list):
        # Embed the fillers
        fillers_embedded = self.filler_embedding(filler_list)

        if self.embed_squeeze:
            fillers_embedded = self.embedding_squeeze_layer(fillers_embedded)

        # Embed the roles
        roles_embedded = self.role_embedding(role_list)

        # Create the sum of the flattened tensor products of the
        # filler and role embeddings
        output = self.sum_layer(fillers_embedded, roles_embedded)

        # If there is a final linear layer to change the output's dimensionality, apply it
        if self.has_last:
            output = self.last_layer(output)

        return output


# At every step of decoding, the decoder is given an input token and
# hidden state. The initial input token is the start-of-string ``<SOS>``
# token, and the first hidden state is the context vector (the encoder's
# last hidden state).
class SCANDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, mytype='LSTM', n_layers=1, dropout_p=0.1):
        # output_size : size of the output language vocabulary
        super(SCANDecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(output_size, hidden_size)
                # output_size is number of words/actions in output vocab.
        self.dropout = nn.Dropout(dropout_p)
            # input_size is number of words in input vocab.
        self.type = mytype
        if self.type == 'GRU':
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers=n_layers, dropout=dropout_p)
        elif self.type == 'LSTM':
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers, dropout=dropout_p)
        elif self.type == 'SRN':
            self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=n_layers, dropout=dropout_p)
        else:
            raise Exception('invalid network type')

        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden):
        # Input
        #  input: a single word index (int)
        #  hidden : (nlayer x 1 x hidden_size)
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
            # embedded is (1 x 1 x hidden_size)
        # embedded = F.relu(embedded)  # Do we need this RELU?
        output, hidden = self.rnn(embedded, hidden)
            # output is variable (1 x 1 x hidden_size)
            # hidden is variable (nlayer x 1 x hidden_size)
            #   The last layer in hidden is the same as output, such that torch.equal(hidden[-1].data,output[0].data))
        netinput = self.out(output[0])
        output = self.softmax(netinput)
        # output is (1 x output_size), which is size of output language vocab
        return output, hidden, netinput


class Elementwise(nn.ModuleList):
    """
    A simple network container.
    Parameters are a list of modules.
    Inputs are a 3d Tensor whose last dimension is the same length
    as the list.
    Outputs are the result of applying modules to inputs elementwise.
    An optional merge parameter allows the outputs to be reduced to a
    single Tensor.
    """

    def __init__(self, merge=None, *args):
        assert merge in [None, 'first', 'concat', 'sum', 'mlp']
        self.merge = merge
        super(Elementwise, self).__init__(*args)

    def forward(self, inputs):
        inputs_ = [feat.squeeze(2) for feat in inputs.split(1, dim=2)]
        assert len(self) == len(inputs_)
        outputs = [f(x) for f, x in zip(self, inputs_)]
        if self.merge == 'first':
            return outputs[0]
        elif self.merge == 'concat' or self.merge == 'mlp':
            return torch.cat(outputs, 2)
        elif self.merge == 'sum':
            return sum(outputs)
        else:
            return outputs


class Embeddings(nn.Module):
    """Words embeddings for encoder/decoder.

    Additionally includes ability to add sparse input features
    based on "Linguistic Input Features Improve Neural Machine Translation"
    :cite:`sennrich2016linguistic`.


    .. mermaid::

       graph LR
          A[Input]
          C[Feature 1 Lookup]
          A-->B[Word Lookup]
          A-->C
          A-->D[Feature N Lookup]
          B-->E[MLP/Concat]
          C-->E
          D-->E
          E-->F[Output]

    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        word_padding_idx (int): padding index for words in the embeddings.
        feat_padding_idx (List[int]): padding index for a list of features
                                   in the embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        feat_vocab_sizes (List[int], optional): list of size of dictionary
            of embeddings for each feature.
        position_encoding (bool): see :class:`~onmt.modules.PositionalEncoding`
        feat_merge (string): merge action for the features embeddings:
            concat, sum or mlp.
        feat_vec_exponent (float): when using `-feat_merge concat`, feature
            embedding size is N^feat_dim_exponent, where N is the
            number of values the feature takes.
        feat_vec_size (int): embedding dimension for features when using
            `-feat_merge mlp`
        dropout (float): dropout probability.
        freeze_word_vecs (bool): freeze weights of word vectors.
    """

    def __init__(self, word_vec_size,
                 word_vocab_size,
                 word_padding_idx,
                 feat_merge='concat',
                 freeze_word_vecs=False):
        self.word_padding_idx = word_padding_idx

        self.word_vec_size = word_vec_size

        # Dimensions and padding for constructing the word embedding matrix
        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        # The embedding matrix look-up tables. The first look-up table
        # is for words. Subsequent ones are for features, if any exist.
        emb_params = zip(vocab_sizes, emb_dims, pad_indices)
        embeddings = [nn.Embedding(vocab, dim, padding_idx=pad)
                      for vocab, dim, pad in emb_params]
        emb_luts = Elementwise(feat_merge, embeddings)

        # The final output size of word + feature vectors. This can vary
        # from the word vector size if and only if features are defined.
        # This is the attribute you should access if you need to know
        # how big your embeddings are going to be.
        self.embedding_size = (word_vec_size)

        # The sequence of operations that converts the input sequence
        # into a sequence of embeddings. At minimum this consists of
        # looking up the embeddings for each word and feature in the
        # input. Model parameters may require the sequence to contain
        # additional operations as well.
        super(Embeddings, self).__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module('emb_luts', emb_luts)

        if freeze_word_vecs:
            self.word_lut.weight.requires_grad = False

    @property
    def word_lut(self):
        """Word look-up table."""
        return self.make_embedding[0][0]

    @property
    def emb_luts(self):
        """Embedding look-up table."""
        return self.make_embedding[0]

    def load_pretrained_vectors(self, emb_file):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
        """

        if emb_file:
            pretrained = torch.load(emb_file)
            pretrained_vec_size = pretrained.size(1)
            if self.word_vec_size > pretrained_vec_size:
                self.word_lut.weight.data[:, :pretrained_vec_size] = pretrained
            elif self.word_vec_size < pretrained_vec_size:
                self.word_lut.weight.data \
                    .copy_(pretrained[:, :self.word_vec_size])
            else:
                self.word_lut.weight.data.copy_(pretrained)

    def forward(self, source, step=None):
        """Computes the embeddings for words and features.

        Args:
            source (LongTensor): index tensor ``(len, batch, nfeat)``

        Returns:
            FloatTensor: Word embeddings ``(len, batch, embedding_size)``
        """
        source = self.make_embedding(source)
        return source


class DecoderRNN(nn.Module):
    def __init__(self, output_size,
                 input_size=1024,
                 hidden_size=512,
                 num_layers=1,
                 dropout_p=0.1,
                 embeddings=None):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size  # Size of the hidden state
        self.output_size = output_size  # Size of the output
        self.input_size = input_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        self.embeddings = embeddings

        for _ in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

        self.dropout = nn.Dropout(dropout_p)

        # Linear layer giving the output
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)  # Softmax layer

    def modify_ckpt_signature(self, decoder_ckpt_path, generator_ckpt_path):
        decoder_ckpt = torch.load(decoder_ckpt_path)
        generator_ckpt = torch.load(generator_ckpt_path)

        embedding_ckpt = collections.OrderedDict()
        embedding_ckpt['embeddings.make_embedding.emb_luts.0.weight'] = \
            decoder_ckpt.pop('embeddings.make_embedding.emb_luts.0.weight')

        decoder_keys = {key: key.split(".", 1)[-1]
                        for key in decoder_ckpt.keys()}
        generator_keys = {key: key.replace("0", "out")
                          for key in generator_ckpt.keys()}

        decoder_ckpt = [(decoder_keys[key], value)
                        for key, value in decoder_ckpt.items()]
        generator_ckpt = [(generator_keys[key], value)
                          for key, value in generator_ckpt.items()]

        decoder_ckpt = collections.OrderedDict(decoder_ckpt)
        decoder_ckpt.update(generator_ckpt)
        decoder_ckpt.update(embedding_ckpt)

        return decoder_ckpt

    # Forward pass
    def forward(self, input_feed, hidden):
        embedded = self.embeddings(input_feed)

        # input feed mechanism. concat the hidden state of last timestep with
        # input emebdding
        # will be lstm_out of TPR encoder [batch_size, embd_dim]
        # split along the seq_length dim
        # decoder_input = hidden[0][0]
        # for emb_t in embedded.split(1):
        #     decoder_input = torch.cat([emb_t.squeeze(0), decoder_input], 1)
        #     decoder_input, hidden = self._run_forward_pass(decoder_input, hidden)

        rnn_output, dec_state = self._run_forward_pass(embedded, hidden)
        rnn_output = self.dropout(rnn_output)

        logits = self.out(rnn_output)
        output = self.softmax(logits)

        return output, hidden, logits

    def _run_forward_pass(self, decoder_input, hidden):
        h_0, c_0 = hidden

        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(decoder_input, (h_0[i], c_0[i]))
            decoder_input = h_1_i
            if i + 1 != self.num_layers:
                decoder_input = self.dropout(decoder_input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return decoder_input, (h_1, c_1)



