# rag_script_with_debug.py
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer, DPRContextEncoder, DPRContextEncoderTokenizerFast, RagRetriever, RagSequenceForGeneration, RagTokenizer
import torch
from datasets import Dataset, Value, Features, Sequence
import os
import json
import faiss
from functools import partial
import warnings
warnings.filterwarnings("ignore")

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the fine-tuned intent classification model and tokenizer
intent_model_path = 'intent_model'
intent_tokenizer = BertTokenizer.from_pretrained(intent_model_path)
intent_model = BertForSequenceClassification.from_pretrained(intent_model_path)
intent_model.to(device)

# Load the CSV file to recreate the label mapping
csv_file_path = 'augmented_wholeIntents.csv'
df = pd.read_csv(csv_file_path)

# Ensure labels are integers
label_mapping = {label: idx for idx, label in enumerate(df['intent'].unique())}
df['labels'] = df['intent'].map(label_mapping)

# Create a reverse label mapping for prediction
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Function to classify intent
def classify_intent(question):
    inputs = intent_tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = intent_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    intent = reverse_label_mapping[predictions.cpu().numpy()[0]]
    return intent

# Load the cleaned and fixed JSON data
with open('cleaned_fixed_yourmom.json', 'r', encoding='utf-8') as file:
    ner_data = json.load(file)

# Convert the data to a Dataset object
data_dict = {
    "text": [item["text"] for item in ner_data],
    "id": [item["id"] for item in ner_data],
    "label": [item["label"] for item in ner_data],
    "title": [item["title"] for item in ner_data]
}

features = Features({
    "text": Value("string"),
    "id": Value("int64"),
    "label": [{'end': Value("int64"), 'labels': Sequence(Value("string")), 'start': Value("int64"), 'text': Value("string")}],
    "title": Value("string")
})

dataset = Dataset.from_dict(data_dict, features=features)

# Initialize context encoder and tokenizer
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base").to(device=device)
ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# Function to embed the text
def embed(documents: dict, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast) -> dict:
    inputs = ctx_tokenizer(
        documents["text"], 
        padding=True, 
        truncation=True, 
        max_length=512,  # Set max_length to handle long text inputs
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    embeddings = ctx_encoder(input_ids, attention_mask=attention_mask, return_dict=True).pooler_output
    
    return {"embeddings": embeddings.detach().cpu().numpy(), "id": documents["id"], "text": documents["text"], "title": documents["title"], "label": documents["label"]}

# Embed the dataset
new_features = Features(
    {"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32")), "id": Value("int64"), "label": [{'end': Value("int64"), 'labels': Sequence(Value("string")), 'start': Value("int64"), 'text': Value("string")}]}
)

print("Embedding the dataset...")
try:
    dataset_tr = dataset.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer),
        batched=True,
        batch_size=4,  # Reduce batch size to avoid CUDA out of memory error
        features=new_features,
    )
    print("Embedding complete.")
except Exception as e:
    print(f"Error during embedding: {e}")

# Save the dataset with embeddings
print("Saving the dataset with embeddings...")
try:
    dataset_tr.save_to_disk('formatted_dataset_with_embeddings')
    print("Dataset saved successfully.")
except Exception as e:
    print(f"Error during dataset save: {e}")

# Load dataset with embeddings
print("Loading dataset with embeddings...")
try:
    dataset_tr = Dataset.load_from_disk('formatted_dataset_with_embeddings')
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error during dataset load: {e}")

# Add Faiss index
print("Adding Faiss index...")
try:
    index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
    dataset_tr.add_faiss_index("embeddings", custom_index=index)
    print("Faiss index added successfully.")
except Exception as e:
    print(f"Error during Faiss index addition: {e}")

# Drop the index before saving the dataset
print("Dropping index before saving the dataset...")
try:
    dataset_tr.drop_index("embeddings")
    dataset_tr.save_to_disk('formatted_dataset_with_index')
    print("Index dropped and dataset saved successfully.")
except Exception as e:
    print(f"Error during index drop or dataset save: {e}")

# Load the indexed dataset
print("Loading the indexed dataset...")
try:
    dataset_tr = Dataset.load_from_disk('formatted_dataset_with_index')
    print("Indexed dataset loaded successfully.")
except Exception as e:
    print(f"Error during indexed dataset load: {e}")

# Re-add the Faiss index
print("Re-adding the Faiss index...")
try:
    index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)  # Recreate the index
    dataset_tr.add_faiss_index("embeddings", custom_index=index)
    print("Faiss index re-added successfully.")
except Exception as e:
    print(f"Error during Faiss index re-addition: {e}")

# Initialize retriever and model
print("Initializing retriever and model...")
try:
    retriever = RagRetriever.from_pretrained(
        "facebook/rag-token-nq", index_name="custom", indexed_dataset=dataset_tr
    )
    rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever).to(device=device)
    rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    print("Retriever and model initialized successfully.")
except Exception as e:
    print(f"Error during retriever or model initialization: {e}")

# Define a function to handle the question
def handle_question(question):
    try:
        intent = classify_intent(question)
        print(f"Intent: {intent}")

        if intent in ["course_info_request", "informational", "majors"]:
            # Use the RAG model for retrieval-based responses
            input_ids = rag_tokenizer.question_encoder(question, return_tensors="pt")["input_ids"].to(device)
            
            print(f"Input IDs for question '{question}': {input_ids}")
            
            contexts = retriever(input_ids)
            print(f"Retrieved Contexts: {contexts}")
            
            generated = rag_model.generate(input_ids, context_input_ids=contexts["context_input_ids"].to(device))
            
            print(f"Generated IDs with context: {generated}")
            
            generated_string = rag_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            print(f"Generated String: {generated_string}")
            
            return generated_string.strip()
        else:
            # Handle other intents accordingly (e.g., greetings, farewells)
            if intent == "greeting":
                return "Hello! How can I assist you today?"
            elif intent == "farewell":
                return "Goodbye! Have a great day!"
            else:
                return "I'm sorry, I didn't understand that. Can you please rephrase?"
    except Exception as e:
        print(f"Error handling question '{question}': {e}")
        return "I'm sorry, I didn't understand that. Can you please rephrase?"

# Example usage
questions = [
    "Who is the instructor of CS 201?",
    "Hi",
    "Who is Albert Levi?",
    "What MAJORS are there"
]

for question in questions:
    response = handle_question(question)
    print(f"Question: {question}")
    print(f"Response: {response}")
