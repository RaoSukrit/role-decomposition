from datasets import load_dataset

import nltk
import torch
import numpy as np
import pandas as pd

import os
import sys

from tqdm import tqdm
from IPython.display import display

sys.path.append(os.path.dirname("../InferSent"))

print(sys.path)
from InferSent.models import InferSent


def load_data():
    snli_dataset = load_dataset("snli")

    snli_premise = list(set(snli_dataset['test']['premise']))

    return snli_premise


def load_encoder(V=1, W2V_PATH='../InferSent/GloVe/glove.840B.300d.txt'):
    MODEL_PATH = f'../InferSent/encoder/infersent{V}.pkl'

    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim':2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}

    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))

    infersent.set_w2v_path(W2V_PATH)

    return infersent


def get_embeddings(encoder, dataset, batch_size):
    encoder.build_vocab(dataset, tokenize=True)

    embeddings = []
    for idx in tqdm(range(0, len(dataset), batch_size), total=len(dataset) // batch_size):
        batch = dataset[idx: idx + batch_size]
        embeddings.append(encoder.encode(batch, tokenize=True))
        # break

    final_embeddings = np.concatenate(embeddings)

    return final_embeddings


def save_output(embeddings, dataset, output_path, batch_size=1024):
    final_df = pd.DataFrame(dataset, columns=['text'])

    final_df['vector'] = embeddings.tolist()
    final_df['vector'] = final_df['vector'].astype(str)

    final_df['vector'] = final_df['vector'].str.replace(',', '', regex=True).str.replace("[", "", regex=True).str.replace("]", "", regex=True)

    display(final_df.head())

    final_df.to_csv(output_path, header=False, index=False, sep='\t')

    print(f"saved final output at {output_path}!")


def main():
    batch_size = 1024
    output_path = "./data/snli_premise.data_from_test"

    dataset = load_data()

    encoder = load_encoder()

    embeddings = get_embeddings(encoder, dataset, batch_size)

    save_output(embeddings, dataset, output_path, batch_size=16)



if __name__ == "__main__":
    main()




