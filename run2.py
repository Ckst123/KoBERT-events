from data_loader import load_data, tokenizer
from models import BertForMultipleLabelSequenceClassification

from transformers import AutoConfig
import torch
from tqdm.auto import tqdm
from transformers import get_scheduler
from transformers import AdamW
from sklearn.metrics import accuracy_score, f1_score


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
        predictions = torch.argmax(logits, dim=-1)
        preds.append(predictions)
        targets.append(batch["labels"])


    preds = torch.cat(preds, dim=0).cpu().numpy()
    targets = torch.cat(targets, dim=0).cpu().numpy()
    N, M = preds.shape
    for i in range(M):
        print("%d results" % (i+1))
        acc = accuracy_score(targets[:,i], preds[:,i])
        f1 = f1_score(targets[:,i], preds[:,i], average='binary')

        print('accuracy', acc * 100)
        print('f1 score', f1 * 100)

    
    


def main():
    checkpoint = "klue/bert-base"
    train_dataloader, eval_dataloader = load_data()
    config = AutoConfig.from_pretrained(checkpoint)
    model = BertForMultipleLabelSequenceClassification.from_pretrained(checkpoint, num_labels=10)
    
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