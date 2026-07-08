import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import evaluate
import numpy as np
import torch
import re
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)

print("Loading dataset...")
df = pd.read_csv('ASLG-PC12_corpus.csv')
print(f"Loaded {len(df)} rows.")

# The dataset has 'gloss' and 'text'
df = df.rename(columns={'gloss': 'input_text', 'text': 'target_text'})

dataset = Dataset.from_pandas(df)
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
print("Dataset split into train and test.")

checkpoint = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokenizer.src_lang = "eng_Latn"
tokenizer.tgt_lang = "eng_Latn"

def preprocess_function(examples):
    inputs = [str(x).lower() for x in examples["input_text"]]
    inputs = [re.sub(r'[^\w\s]', '', x) for x in inputs]
    
    targets = [str(x).lower() for x in examples["target_text"]]
    targets = [re.sub(r'[^\w\s]', '', x) for x in targets]
    
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizing dataset...")
tokenized_datasets = split_dataset.map(preprocess_function, batched=True, remove_columns=split_dataset["train"].column_names)

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

training_args = Seq2SeqTrainingArguments(
    output_dir="./asl-english-transformer",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4, 
    optim="adafactor",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    weight_decay=1e-3,
    save_total_limit=2,
    num_train_epochs=5, 
    predict_with_generate=True,
    generation_max_length=128,
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    report_to="none",
    logging_steps=50
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics
)

print("Starting training...")
try:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        
    trainer.train()
    model.save_pretrained("./final_asl_english_model")
    print("Training Complete! Model saved to final_asl_english_model folder.")
except Exception as e:
    print(f"An error occurred during training: {e}")
