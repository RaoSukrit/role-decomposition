import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torchtext.vocab import Vocab

import collections
import json

from models import DecoderRNN, Embeddings, Elementwise

from torchtext.vocab import Vocab


def build_embeddings(opt, vocab):
    """
    Args:
        opt: the option in current environment.
        text_field(TextMultiField): word and feats field.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    emb_dim = opt['tgt_word_vec_size']

    word_padding_idx = vocab.stoi['<pad>']
    num_word_embeddings = len(vocab.stoi)

    print(f"num_word_embeddings={num_word_embeddings}")

    freeze_word_vecs = opt['freeze_word_vecs_dec']

    emb = Embeddings(
        word_vec_size=emb_dim,
        word_vocab_size=num_word_embeddings,
        word_padding_idx=word_padding_idx,
        feat_merge=opt['feat_merge'],
        freeze_word_vecs=freeze_word_vecs
    )
    return emb


def build_decoder(opt, embeddings):
    """
    Various decoder dispatcher function.
    Args:
        opt: the option in current environment.
        embeddings (Embeddings): vocab embeddings for this decoder.
    """
    decoder_model = DecoderRNN(opt['output_size'],
                               input_size=opt['input_size'],
                               hidden_size=opt['hidden_size'],
                               num_layers=opt['num_layers'],
                               embeddings=embeddings)

    decoder_ckpt = \
        decoder_model.modify_ckpt_signature(opt['decoder_ckpt_path'],
                                            opt['generator_ckpt_path'])
    decoder_model.load_state_dict(decoder_ckpt)

    return decoder_model


def build_decoder_with_embeddings(model_opt, vocab=None):
    # Build embeddings
    tgt_embd = build_embeddings(model_opt, vocab)

    # Build decoder.
    model_opt['output_size'] = len(vocab.stoi)
    decoder = build_decoder(model_opt, tgt_embd)

    return decoder


def build_vocab(src_vocab_path, tgt_vocab_path):
    with open(src_vocab_path, 'rb') as fh:
        src_vocab = json.load(fh)

    with open(tgt_vocab_path, 'rb') as fh:
        tgt_vocab = json.load(fh)

    src_vocab['<pad>'] = src_vocab.pop("<blank>")
    tgt_vocab['<pad>'] = tgt_vocab.pop("<blank>")

    src_vocab = Vocab(src_vocab)
    tgt_vocab = Vocab(tgt_vocab)

    return src_vocab, tgt_vocab
