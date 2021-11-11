from datasets import load_dataset
from transformers import DataCollatorWithPadding, AutoTokenizer, DataCollatorForTokenClassification, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from numpy.random import randint

checkpoint = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
label_binary = {'+': 1, '-':1, '1':1, '0': 0, 1:1, 0:0}
label_4class = {'+': 3, '-':2, '1':1, '0': 0, 1:1, 0:0}
label_3class = {'+': 2, '-':1, '0': 0, 0:0}

label_list = ['news_headline','article','확진자수','완치자수','사망여부','집단감염','백신관련','방역지침','경제지원','마스크','국제기구','병원관련']
num_classes = {'확진자수':4,'완치자수':4,'사망여부':4,'집단감염':3,'백신관련':4,'방역지침':4,'경제지원':2,'마스크':4,'국제기구':4,'병원관련':4}
to_labels = {2:label_binary, 3:label_3class, 4:label_4class}


def load_data():
    raw_datasets = load_dataset('csv', data_files={'train': 'train.csv', 'validation': 'test.csv'})
    print(raw_datasets)

    def tokenize_function(example):
        return tokenizer(example['news_headline'], example["article"], truncation=True)

    def trim(example):
        example['article'] = example['article'].strip().replace("{", " ").replace("}", " ").replace("[", " ").replace("]", " ").replace('","', " ").strip()
        return example
    

    def get_labels(example):
        example['labels'] = [label_binary[example[lab]] for lab in label_list[2:]]
        return example

    tokenized_datasets = raw_datasets.map(trim).map(tokenize_function, batched=True).map(get_labels)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(
        label_list
    )
    tokenized_datasets.set_format("torch")
    
    print(tokenized_datasets["train"].column_names)
    print(tokenized_datasets["train"][0])
    print(tokenizer.convert_ids_to_tokens(tokenized_datasets["train"][0]["input_ids"]))


    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )

    return train_dataloader, eval_dataloader


def load_data_with_label_name(labels):
    raw_datasets = load_dataset('csv', data_files={'train': 'train.csv', 'validation': 'test.csv'})
    print(raw_datasets)

    def tokenize_function(example):
        return tokenizer(example['news_headline'], example["article"], truncation=True)

    def trim(example):
        example['article'] = example['article'].strip().replace("{", " ").replace("}", " ").replace("[", " ").replace("]", " ").replace('","', " ").strip()
        return example
    

    def get_labels(example):
        example['labels'] = [label_binary[example[lab]] for lab in labels]
        return example

    tokenized_datasets = raw_datasets.map(trim).map(tokenize_function, batched=True).map(get_labels)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(
        label_list
    )
    tokenized_datasets.set_format("torch")
    
    print(tokenized_datasets["train"].column_names)
    print(tokenized_datasets["train"][0])
    print(tokenizer.convert_ids_to_tokens(tokenized_datasets["train"][0]["input_ids"]))


    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )

    return train_dataloader, eval_dataloader

def load_data_with_total_binary():
    raw_datasets = load_dataset('csv', data_files={'train': 'train.csv', 'validation': 'test.csv'})
    print(raw_datasets)

    def tokenize_function(example):
        return tokenizer(example['news_headline'], example["article"], truncation=True)

    def trim(example):
        example['article'] = example['article'].strip().replace("{", " ").replace("}", " ").replace("[", " ").replace("]", " ").replace('","', " ").strip()
        return example
    

    def get_labels(example):
        example['labels'] = [np.minimum(sum([label_binary[example[lab]] for lab in label_list[2:]]), 1.)]
        return example

    tokenized_datasets = raw_datasets.map(trim).map(tokenize_function, batched=True).map(get_labels)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(
        label_list
    )
    tokenized_datasets.set_format("torch")
    
    print(tokenized_datasets["train"].column_names)
    print(tokenized_datasets["train"][0])
    print(tokenizer.convert_ids_to_tokens(tokenized_datasets["train"][0]["input_ids"]))


    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )

    return train_dataloader, eval_dataloader

def load_data_multiclass():
    raw_datasets = load_dataset('csv', data_files={'train': 'train.csv', 'validation': 'test.csv'})
    print(raw_datasets)

    def tokenize_function(example):
        return tokenizer(example['news_headline'], example["article"], truncation=True)

    def trim(example):
        example['article'] = example['article'].strip().replace("{", " ").replace("}", " ").replace("[", " ").replace("]", " ").replace('","', " ").strip()
        return example
    

    def get_labels(example):
        example['labels'] = [to_labels[num_classes[lab]][example[lab]] for lab in label_list[2:]]
        return example

    tokenized_datasets = raw_datasets.map(trim).map(tokenize_function, batched=True).map(get_labels)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(
        label_list
    )
    tokenized_datasets.set_format("torch")
    
    print(tokenized_datasets["train"].column_names)
    print(tokenized_datasets["train"][0])
    print(tokenizer.convert_ids_to_tokens(tokenized_datasets["train"][0]["input_ids"]))


    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )

    return train_dataloader, eval_dataloader


def load_data_multiclass_with_label_name(task):
    raw_datasets = load_dataset('csv', data_files={'train': 'train.csv', 'validation': 'test.csv'})
    print(raw_datasets)

    def tokenize_function(example):
        return tokenizer(example['news_headline'], example["article"], truncation=True)

    def trim(example):
        example['article'] = example['article'].strip().replace("{", " ").replace("}", " ").replace("[", " ").replace("]", " ").replace('","', " ").strip()
        return example
    

    def get_labels(example):
        example['labels'] = [to_labels[num_classes[lab]][example[lab]] for lab in task]
        return example

    tokenized_datasets = raw_datasets.map(trim).map(tokenize_function, batched=True).map(get_labels)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_datasets = tokenized_datasets.remove_columns(
        label_list
    )
    tokenized_datasets.set_format("torch")
    
    print(tokenized_datasets["train"].column_names)
    print(tokenized_datasets["train"][0])
    print(tokenizer.convert_ids_to_tokens(tokenized_datasets["train"][0]["input_ids"]))


    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )

    return train_dataloader, eval_dataloader


if __name__ == '__main__':
    train_dataloader, eval_dataloader = load_data_multiclass()