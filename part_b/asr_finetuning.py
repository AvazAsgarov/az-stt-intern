# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
import librosa
import jiwer

from torch.utils.data import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

# =========================================================
# CONFIG
# =========================================================

CONFIG = {
    "model_name": "openai/whisper-small",
    "data_root": "../az", # Updated path to point to the 'az' folder in the root
    "audio_dir": "clips",
    "train_tsv": "train.tsv",
    "val_tsv": "dev.tsv",
    "sr": 16000,
    "max_train": 200,
    "max_val": 50,
    "output_dir": "../results/checkpoints", # Updated to save in results directory
    "best_model_dir": "../results/best_model"
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================

def load_data(tsv_path, audio_dir, n):
    df = pd.read_csv(tsv_path, sep="\t")[["path", "sentence"]].dropna()
    df["file"] = df["path"].apply(lambda x: os.path.join(audio_dir, x))
    df = df[df["file"].apply(os.path.exists)]
    return df.head(n).reset_index(drop=True)

# =========================================================
# DATASET
# =========================================================

class ASRDataset(Dataset):
    def __init__(self, df, processor, sr):
        self.df = df
        self.processor = processor
        self.sr = sr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio, _ = librosa.load(row["file"], sr=self.sr)
        inputs = self.processor.feature_extractor(
            audio,
            sampling_rate=self.sr
        ).input_features[0]
        labels = self.processor.tokenizer(row["sentence"]).input_ids
        return {
            "input_features": inputs,
            "labels": labels
        }

# =========================================================
# COLLATOR
# =========================================================

class WhisperCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        input_features = [{"input_features": x["input_features"]} for x in batch]
        label_features = [{"input_ids": x["labels"]} for x in batch]
        batch_inputs = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"] == 0,
            -100
        )
        batch_inputs["labels"] = labels
        return batch_inputs

# =========================================================
# METRICS
# =========================================================

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    preds = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    refs = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {
        "wer": jiwer.wer(refs, preds),
        "cer": jiwer.cer(refs, preds)
    }

# =========================================================
# MODEL + PROCESSOR
# =========================================================

processor = WhisperProcessor.from_pretrained(CONFIG["model_name"])
model = WhisperForConditionalGeneration.from_pretrained(CONFIG["model_name"])

model.config.use_cache = False
model.generation_config.forced_decoder_ids = None

# =========================================================
# DATA PREP
# =========================================================

audio_dir = os.path.join(CONFIG["data_root"], CONFIG["audio_dir"])
train_df = load_data(
    os.path.join(CONFIG["data_root"], CONFIG["train_tsv"]),
    audio_dir,
    CONFIG["max_train"]
)
val_df = load_data(
    os.path.join(CONFIG["data_root"], CONFIG["val_tsv"]),
    audio_dir,
    CONFIG["max_val"]
)

train_ds = ASRDataset(train_df, processor, CONFIG["sr"])
val_ds = ASRDataset(val_df, processor, CONFIG["sr"])
collator = WhisperCollator(processor)

# =========================================================
# TRAINING ARGS
# =========================================================

training_args = Seq2SeqTrainingArguments(
    output_dir=CONFIG["output_dir"],
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-5,
    max_steps=100,
    eval_steps=25,
    save_steps=25,
    eval_strategy="steps",
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# =========================================================
# TRAINER
# =========================================================

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=collator,
    compute_metrics=compute_metrics
)

# =========================================================
# TRAIN
# =========================================================

print("Starting Training...")
trainer.train()

# Save best model
trainer.save_model(CONFIG["best_model_dir"])
processor.save_pretrained(CONFIG["best_model_dir"])

print("\n TRAINING COMPLETE")

# =========================================================
# EVALUATION (BASE vs FINE-TUNED)
# =========================================================

def evaluate_model(eval_model, df):
    eval_model.eval()
    wers, cers = [], []
    for _, row in df.iterrows():
        audio, _ = librosa.load(row["file"], sr=CONFIG["sr"])
        inputs = processor(
            audio,
            sampling_rate=CONFIG["sr"],
            return_tensors="pt"
        ).input_features
        with torch.no_grad():
            pred = eval_model.generate(inputs)
        hyp = processor.batch_decode(pred, skip_special_tokens=True)[0]
        ref = row["sentence"]
        wers.append(jiwer.wer(ref, hyp))
        cers.append(jiwer.cer(ref, hyp))
    return np.mean(wers), np.mean(cers)

# =========================================================
# RESULTS COMPARISON
# =========================================================

base_model = WhisperForConditionalGeneration.from_pretrained(CONFIG["model_name"])
ft_model = WhisperForConditionalGeneration.from_pretrained(CONFIG["best_model_dir"])

base_wer, base_cer = evaluate_model(base_model, val_df)
ft_wer, ft_cer = evaluate_model(ft_model, val_df)

print("\n================ RESULTS ================")
print(f"Base Model      \u2192 WER: {base_wer:.4f} | CER: {base_cer:.4f}")
print(f"Fine-Tuned Model\u2192 WER: {ft_wer:.4f} | CER: {ft_cer:.4f}")
print("=========================================")
