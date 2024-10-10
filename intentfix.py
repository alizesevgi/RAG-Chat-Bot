# fine_tune_intent_classification.py
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
import torch
from datasets import Dataset
import json
# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the augmented CSV file
csv_file_path = 'augmented_wholeIntents.csv'
df = pd.read_csv(csv_file_path)

# Ensure labels are integers
label_mapping = {label: idx for idx, label in enumerate(df['intent'].unique())}
df['labels'] = df['intent'].map(label_mapping)

# Save the label mapping to ensure consistency
label_mapping_path = 'label_mapping.json'
with open(label_mapping_path, 'w') as f:
    json.dump(label_mapping, f)

# Convert DataFrame to Dataset
dataset = Dataset.from_pandas(df)

# Tokenize the dataset with padding
intent_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize(batch):
    return intent_tokenizer(batch['patterns'], padding='max_length', truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Split the dataset into train and validation sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
val_dataset = train_test_split['test']

# Set the format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Fine-tune the model for intent classification
model_name = 'bert-base-uncased'
intent_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label_mapping))
intent_model.to(device)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=intent_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("Starting training...")
trainer.train()
print("Training complete.")

# Save the fine-tuned model
intent_model.save_pretrained('intent_model')
intent_tokenizer.save_pretrained('intent_model')

print("Fine-tuned intent classification model saved to 'intent_model'.")
