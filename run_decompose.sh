#!/bin/bash

export data_path=$1
export data_prefix=$2
export output_dir=$3

export num_roles=50
export role_dim=50
export filler_dim=512
export vocab_size=745
export hidden_size=512

export cogs_src_vocab=""
export cogs_tgt_vocab=""
export decoder_ckpt_path=""
export generator_ckpt_path=""

export decoder="cogs"
export test_decoder="True"
export decoder_input_size=$hidden_size
export decoder_hidden_size=512
export decoder_num_layers=1
export tgt_word_vec_size=512
export decoder_feat_merge="concat"

export patience=10


python3 decompose.py --data_path=$data_path \
                     --data_prefix=$data_prefix \
                     --output_dir=$output_dir \
                     --num_roles=$num_roles \
                     --filler_dim=$filler_dim \
                     --role_dim=$role_dim \
                     --role_learning \
                     --vocab_size=$vocab_size \
                     --hidden_size=$hidden_size \
                     --cogs_src_vocab=$cogs_src_vocab \
                     --cogs_tgt_vocab=$cogs_tgt_vocab \
                     --test_decoder=$test_decoder \
                     --decoder=$decoder \
                     --decoder_input_size=$decoder_input_size \
                     --decoder_hidden_size=$decoder_hidden_size \
                     --decoder_num_layers=$decoder_num_layers \
                     --tgt_word_vec_size=$tgt_word_vec_size \
                     --decoder_feat_merge=$decoder_feat_merge \
                     --freeze_word_vecs_dec \
                     --patience=$patience
