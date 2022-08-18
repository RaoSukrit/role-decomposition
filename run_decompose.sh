#!/bin/bash

export data_path=$1
export data_prefix=$2
export output_dir=$3

export num_roles=50
export role_dim=50
export filler_dim=512
export vocab_size=745
export hidden_size=512

export cogs_src_vocab="/Users/sukritrao/Documents/NYU/Coursework/Summer2022/Research/role-analysis/COGS/exp_data/1_example/1_example_src_vocab.json"
export cogs_tgt_vocab="/Users/sukritrao/Documents/NYU/Coursework/Summer2022/Research/role-analysis/COGS/exp_data/1_example/1_example_tgt_vocab.json"
export decoder_ckpt_path="/Users/sukritrao/Documents/NYU/Coursework/Summer2022/Research/role-analysis/COGS/src/OpenNMT-py/tf_checkpoints/1_example_lstm_uni_no_att_1layers/decoder/decoder.weights"
export generator_ckpt_path="/Users/sukritrao/Documents/NYU/Coursework/Summer2022/Research/role-analysis/COGS/src/OpenNMT-py/tf_checkpoints/1_example_lstm_uni_no_att_1layers/decoder/generator.weights"

export decoder="cogs"
export test_decoder="True"
export decoder_input_size=1024
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
                     --decoder_ckpt_path=$decoder_ckpt_path \
                     --generator_ckpt_path=$generator_ckpt_path \
                     --decoder_input_size=$decoder_input_size \
                     --decoder_hidden_size=$decoder_hidden_size \
                     --decoder_num_layers=$decoder_num_layers \
                     --decoder_word_vec_size=$tgt_word_vec_size \
                     --decoder_feat_merge=$decoder_feat_merge \
                     --decoder_freeze_word_embd \
                    #  --train="False" \
                     --patience=$patience
