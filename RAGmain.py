import os
import json
import faiss
import torch
from datasets import Value, Features, Sequence, Dataset
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast, RagRetriever, RagSequenceForGeneration, RagTokenizer
from functools import partial
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

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

# Verify the structure of the data
print(f"Data Dictionary Keys: {data_dict.keys()}")
print(f"First Item: {data_dict['text'][0]}")

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

# Verify the columns
print(f"Dataset Features: {dataset.features}")
print(f"First Dataset Item: {dataset[0]}")

# Ensure new features structure
new_features = Features(
    {"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32")), "id": Value("int64"), "label": [{'end': Value("int64"), 'labels': Sequence(Value("string")), 'start': Value("int64"), 'text': Value("string")}]}
)

# Reduce batch size to lower memory requirements
dataset_tr = dataset.map(
    partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=ctx_tokenizer),
    batched=True,
    batch_size=4,  # Reduce batch size to avoid CUDA out of memory error
    features=new_features,
)

# Save the dataset with embeddings
dataset_tr.save_to_disk('formatted_dataset_with_embeddings')

# Load dataset with embeddings
dataset_tr = Dataset.load_from_disk('formatted_dataset_with_embeddings')

# Add Faiss index
index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
dataset_tr.add_faiss_index("embeddings", custom_index=index)

# Drop the index before saving the dataset
dataset_tr.drop_index("embeddings")
dataset_tr.save_to_disk('formatted_dataset_with_index')

# Load the indexed dataset
dataset_tr = Dataset.load_from_disk('formatted_dataset_with_index')

# Re-add the Faiss index
index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)  # Recreate the index
dataset_tr.add_faiss_index("embeddings", custom_index=index)

# Initialize retriever and model
retriever = RagRetriever.from_pretrained(
    "facebook/rag-token-nq", index_name="custom", indexed_dataset=dataset_tr
)
rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever).to(device=device)
rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")

# Load the fine-tuned intent classification model and tokenizer
intent_model_path = 'intent_model'
intent_tokenizer = BertTokenizer.from_pretrained(intent_model_path)
intent_model = BertForSequenceClassification.from_pretrained(intent_model_path)
intent_model.to(device)

# Load the label mapping to ensure consistency
label_mapping_path = 'label_mapping.json'
with open(label_mapping_path, 'r') as f:
    label_mapping = json.load(f)

# Create a reverse label mapping for prediction
reverse_label_mapping = {v: k for k, v in label_mapping.items()}

def classify_intent(question):
    inputs = intent_tokenizer(question, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = intent_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    intent = reverse_label_mapping[predictions.cpu().numpy()[0]]
    
    # Debug print
    print(f"Question: {question}")
    print(f"Predicted Intent: {intent}")
    
    return intent

# Define a function to handle the question
def handle_question(question):
    intent = classify_intent(question)
    print(f"Intent: {intent}")

    # List of intents that should trigger the RAG model response
    rag_intents = ["course_info_request", "informational", "majors", "COURSE", "request_person_info", "PER", 
                   "Exam", "fees", "creator", "hostel", "transportation", "programs", "Bachelor", 
                   "engineering", "enrollment", "payments_1", "credit"]

    if intent in rag_intents:
        # Use the RAG model for retrieval-based responses
        input_ids = rag_tokenizer.question_encoder(question, return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(device)  # Move input IDs to the correct device
        generated = rag_model.generate(input_ids, max_length=50)  # Increase max_length if necessary
        generated_string = rag_tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        print(f"Generated Response: {generated_string.strip()}")  # Debug print
        return generated_string.strip()
    else:
        # Handle other intents accordingly (e.g., greetings, farewells)
        if intent == "greeting":
            return "Hello! How can I assist you today?"
        elif intent == "farewell":
            return "Goodbye! Have a great day!"
        elif intent == "thanks":
            return "You're welcome!"
        else:
            return "I'm not sure how to respond to that."

# Example usage
question = "Who is the instructor of CS 201?"
response = handle_question(question)
print(f"Question: {question}")
print(f"Response: {response}")

question = "Hi"
response = handle_question(question)
print(f"Question: {question}")
print(f"Response: {response}")

question = "Who is Albert Levi?"
response = handle_question(question)
print(f"Question: {question}")
print(f"Response: {response}")

question = "What MAJORS are there"
response = handle_question(question)
print(f"Question: {question}")
print(f"Response: {response}")
