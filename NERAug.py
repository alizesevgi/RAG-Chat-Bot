# data_augmentation.py
import pandas as pd
from transformers import pipeline

# Load the CSV file for data augmentation
csv_file_path = 'wholeIntents.csv'
df = pd.read_csv(csv_file_path)

# Ensure labels are integers
label_mapping = {label: idx for idx, label in enumerate(df['intent'].unique())}
df['labels'] = df['intent'].map(label_mapping)

# Data augmentation using a paraphrasing model
paraphrase_pipeline = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

def augment_data(text, num_augments=2):
    augments = paraphrase_pipeline(text, num_return_sequences=num_augments, num_beams=5)
    return [augment['generated_text'] for augment in augments]

augmented_data = []
for index, row in df.iterrows():
    augmented_texts = augment_data(row['patterns'])
    for text in augmented_texts:
        augmented_data.append({"intent": row['intent'], "patterns": text, "labels": row['labels']})

augmented_df = pd.DataFrame(augmented_data)

# Combine original and augmented data
combined_df = pd.concat([df, augmented_df]).reset_index(drop=True)

# Save the augmented data to a CSV file
combined_df.to_csv('augmented_wholeIntents.csv', index=False)

print("Data augmentation complete. Augmented data saved to 'augmented_wholeIntents.csv'.")
