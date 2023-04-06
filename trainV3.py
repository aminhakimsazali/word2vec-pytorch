import argparse
import yaml
import os
import torch
import torch.nn as nn

from utils.dataloader import get_dataloader_and_vocab
from utils.trainer import Trainer
from utils.helper import (
    get_model_class,
    get_optimizer_class,
    get_lr_scheduler,
    save_config,
    save_vocab,
)

from utils.dataloader import collate_skipgram, collate_cbow, build_vocab
from functools import partial
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from sklearn.utils import shuffle



class MalaysiaKiniData(Dataset):
    def __init__(self, data):
        #load data
        self.data = data
        
    def __getitem__(self, index):
        #return exact data
        return self.data[index]
    
    def __len__(self):
        #return data length
        return len(self.data)


def get_dataloader_and_vocab_malay_dataset(
    model_name, data, batch_size, shuffle, vocab=None,
):
    
    """
    function use data as parameter
    """

    tokenizer = get_tokenizer("basic_english")

    def yield_sentences():
        for sent in data:
            yield sent

    sentences_generator = yield_sentences()

    data_iter = MalaysiaKiniData(data)
    tokenizer = get_tokenizer("basic_english")

    if not vocab:
        vocab = build_vocab(sentences_generator, tokenizer)
        
    text_pipeline = lambda x: vocab(tokenizer(x))

    if model_name == "cbow":
        collate_fn = collate_cbow
    elif model_name == "skipgram":
        collate_fn = collate_skipgram
    else:
        raise ValueError("Choose model from: cbow, skipgram")
    print("Vocab size: ", len(vocab.get_stoi()))
    dataloader = DataLoader(
        data_iter,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, text_pipeline=text_pipeline),
    )
    return dataloader, vocab

def train(config):
    torch.cuda.set_device(0)
    # torch.cuda.init()
    # handle = torch.cuda.cublas_initialize()


    os.makedirs(config["model_dir"])
    #read the data and combine column Target and Source into 1 Pandas Series
    # df = pd.read_excel("data/malaysia-kini/06_MalaysiaKini & Awani Articles 2022.xlsx")
    df = pd.read_excel("data/malaysia-kini/07_MalaysiaKini & Awani Articles 2022 + OSCAR.xlsx")
    df.columns = ['Text']

    # all_data = df['Target'].tolist()
    all_data = df['Text'].tolist()
    # all_data.extend( df['Source'].tolist())
    dfSeries = pd.Series(all_data)

    dfSeries = dfSeries[dfSeries.str.len() > 3]
    #preprocessing to drop NA and duplications row
    dfSeries.drop_duplicates(inplace=True)
    dfSeries.dropna(inplace=True)
    
    #randomise the dataset
    dfSeries = shuffle(dfSeries)
    #reset index after performing drop duplicates and dropNA
    dfSeries.reset_index(drop=True, inplace=True)
    
    #Load Validation dataset
    train_dataloader, vocab = get_dataloader_and_vocab_malay_dataset(
    model_name=config["model_name"],
    data = dfSeries,
    batch_size=config["train_batch_size"],
    shuffle=False,
    vocab=None)

    #Load Validation dataset
    val_dataloader, _ = get_dataloader_and_vocab_malay_dataset(
        model_name=config["model_name"],
        data = dfSeries,
        batch_size=config["train_batch_size"],
        shuffle=config["shuffle"],
        vocab=None)

    vocab_size = len(vocab.get_stoi())
    print(f"Vocabulary size: {vocab_size}")

    model_class = get_model_class(config["model_name"])
    model = model_class(vocab_size=vocab_size)
    criterion = nn.CrossEntropyLoss()

    optimizer_class = get_optimizer_class(config["optimizer"])
    optimizer = optimizer_class(model.parameters(), lr=config["learning_rate"])
    lr_scheduler = get_lr_scheduler(optimizer, config["epochs"], verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA Device : {device}")

    trainer = Trainer(
        model=model,
        epochs=config["epochs"],
        train_dataloader=train_dataloader,
        train_steps=config["train_steps"],
        val_dataloader=val_dataloader,
        val_steps=config["val_steps"],
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_frequency=config["checkpoint_frequency"],
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=config["model_dir"],
        model_name=config["model_name"],
    )

    trainer.train()
    print("Training finished.")

    trainer.save_model()
    trainer.save_loss()
    save_vocab(vocab, config["model_dir"])
    save_config(config, config["model_dir"])
    print("Model artifacts saved to folder:", config["model_dir"])
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to yaml config')
    args = parser.parse_args()
    
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    train(config)