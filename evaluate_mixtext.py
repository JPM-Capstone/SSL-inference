import os
import sys
import pickle
import numpy as np
from glob import glob
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaForSequenceClassification
from torch.utils.data import DataLoader
from data import LabeledDataset
from mixtext import MixText
from collections import OrderedDict

BATCH_SIZE = 32
PAD_token = 1 # RoBERTa

PATH_TO_RESULTS = os.path.join("..", "Mix-Text", "results")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(model, val_loader):

    model.eval()
    val_acc = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch[0].to(DEVICE)
            labels = batch[2].to(DEVICE)

            outputs = model(input_ids)
            val_acc += (outputs.logits.argmax(dim=1) == labels).sum().item()

    return val_acc/len(val_loader.dataset)


def main(config_name):

    runs = glob(os.path.join(PATH_TO_RESULTS, config_name, "run_*"))

    logger = open(os.path.join("mixtext_results", f"{config_name}.csv"), 'w')
    logger.write("Run, Test Accuracy\n")

    for i, run in enumerate(runs):
        for ckpt in glob(os.path.join(run, "*.pt")):

            # Load the pretrained model
            model = MixText(10)

            state_dict = torch.load(ckpt)
            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                name = k[len('module.'):] # remove `module.`
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)
            model.to(DEVICE)

            # Creating training and validation datasets
            test_dataset = LabeledDataset()
            
            test_loader = DataLoader(test_dataset, 
                                    batch_size = BATCH_SIZE, 
                                    shuffle=False,
                                    collate_fn = collate_batch)

            accuracy = evaluate(model, test_loader)

            logger.write(f"{i + 1}, {accuracy:.6f}\n")

    logger.close()
    
def collate_batch(batch):
    """
    Labeled batch: input_ids, attention_mask, labels
    """

    input_ids, attention_mask, labels = [], [], []
    for (_input, _mask, _label) in batch:
        input_ids.append(_input)
        attention_mask.append(_mask)
        labels.append(_label)
    
    input_ids = pad_sequence(input_ids, batch_first = True, padding_value = PAD_token)
    attention_mask = pad_sequence(attention_mask, batch_first = True, padding_value = 0)
            
    return input_ids, attention_mask, torch.tensor(labels)
    
    
if __name__ == '__main__':
    
    main(sys.argv[1])