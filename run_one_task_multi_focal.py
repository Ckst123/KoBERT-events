from data_loader import load_data_multiclass_with_label_name, tokenizer
from models import BertForMultipleSequenceClassificationWithFocalLoss

from transformers import AutoConfig
import torch
from tqdm.auto import tqdm
from transformers import get_scheduler
from transformers import AdamW
from sklearn.metrics import accuracy_score, f1_score

label_list = ['확진자수','완치자수','사망여부','집단감염','백신관련','방역지침','경제지원','마스크','국제기구','병원관련']
num_classes = {'확진자수':4,'완치자수':4,'사망여부':4,'집단감염':3,'백신관련':4,'방역지침':4,'경제지원':2,'마스크':4,'국제기구':4,'병원관련':4}

import sys

task = sys.argv[1:] if len(sys.argv) > 1 else ['확진자수']


def train(model, optimizer, lr_scheduler, train_dataloader, num_epochs, num_training_steps, device):
    
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)


def eval(model, eval_dataloader, metric, device):
    model.eval()
    preds = []
    targets = []
    probs = []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        logits = outputs.logits
        predictions = torch.stack([torch.argmax(logit, dim=-1) for logit in logits], dim=1)
        preds.append(predictions)
        targets.append(batch["labels"])


    preds = torch.cat(preds, dim=0).cpu().numpy()
    targets = torch.cat(targets, dim=0).cpu().numpy()
    N, M = preds.shape
    for i in range(M):
        print("%s results" % task[i])
        acc = accuracy_score(targets[:,i], preds[:,i])
        f1 = f1_score(targets[:,i], preds[:,i], average='macro')

        print('accuracy', acc * 100)
        print('macro f1 score', f1 * 100)

    
    


def main():
    checkpoint = "klue/bert-base"
    train_dataloader, eval_dataloader = load_data_multiclass_with_label_name(task)
    config = AutoConfig.from_pretrained(checkpoint)
    config.num_classes=[num_classes[lab] for lab in task]
    model = BertForMultipleSequenceClassificationWithFocalLoss.from_pretrained(checkpoint, config=config)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)


    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    train(model, optimizer, lr_scheduler, train_dataloader, num_epochs, num_training_steps, device)
    print()

    eval(model, eval_dataloader, 'metric', device)
    

if __name__ == '__main__':
    main()