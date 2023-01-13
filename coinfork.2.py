#0.7457928657531738
import yfinance as yf
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.data.sampler import RandomSampler
import torch
from transformers import get_linear_schedule_with_warmup
from tqdm import trange

# Fetch historical data for Fantom
ftm = yf.Ticker("FTM-USD")
data = ftm.history(period="max")

# Preprocess the data to extract relevant information for your classification task
# This will likely involve cleaning and transforming the data
# You can use pandas to do this

# Load the pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Prepare your labeled dataset of sentences and their corresponding labels
# You can use any dataset of your choice, this is just an example
sentences = ["This is a positive sentence.", "This is a negative sentence.", "This is a neutral sentence."]
labels = [1, 0, 2]  # 1 for positive, 0 for negative, 2 for neutral

# Convert the dataset to input format for BERT
input_ids = [tokenizer.encode(sent, add_special_tokens=True) for sent in sentences]
attention_masks = [[float(i>0) for i in sent] for sent in input_ids]
labels = torch.tensor(labels)

# Split the dataset into training and validation sets
train_data, val_data = random_split(TensorDataset(torch.tensor(input_ids).long(), torch.tensor(attention_masks).long(), labels), [int(0.8 * len(input_ids)), len(input_ids) - int(0.8 * len(input_ids))])

batch_size = 16  # Assign a value for batch_size
num_train_epochs = 10  # Assign a value for number of training epochs
device = "cpu"  # Assign a value for device, can be "cpu" or "cuda"

# Create an iterator of our data with torch DataLoader
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
val_dataloader = DataLoader(val_data, batch_size=batch_size)

# Define the optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=-1)

# Train the model
model.train()
for _ in trange(int(num_train_epochs), desc="Epoch"):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # Add batch to device
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_attention_masks, b_labels = batch
        # Clear out the gradients (by default they accumulate)
        optimizer.zero_grad()
        # Forward pass
        outputs = model(b_input_ids, attention_mask=b_attention_masks, labels=b_labels)
        loss = outputs[0]
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()
        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
    print("Train loss: {}".format(tr_loss/nb_tr_steps))

# Evaluation loop
model.eval()
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
for batch in val_dataloader:
    # Add batch to device
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_attention_masks, b_labels = batch
    # Forward pass
    with torch.no_grad():
        outputs = model(b_input_ids, attention_mask=b_attention_masks, labels=b_labels)
        tmp_eval_loss, logits = outputs[:2]
        logits = logits.detach().cpu().numpy()
        predictions = np.argmax(logits, axis=1)
        eval_accuracy += accuracy_score(b_labels, predictions)
    nb_eval_steps += 1
print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
