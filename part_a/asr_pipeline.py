# -*- coding: utf-8 -*-
import os
import re
import logging
import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import jiwer
from tqdm import tqdm
import librosa

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

CONFIG = {
    "model_name": "openai/whisper-medium",
    "data_root": "../az", # Root relative path
    "metadata_file": "test.tsv",
    "audio_subdir": "clips",
    "max_samples": 500,
    "batch_size": 4,
    "target_sampling_rate": 16000,
    "force_language": "az"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def normalize_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_metadata(config):
    metadata_path = os.path.join(config["data_root"], config["metadata_file"])
    df = pd.read_csv(metadata_path, sep="\t", dtype=str)
    df = df[["path", "sentence"]].dropna()
    df["audio_path"] = df["path"].apply(
        lambda p: os.path.join(config["data_root"], config["audio_subdir"], os.path.basename(p))
    )
    df = df[df["audio_path"].apply(os.path.isfile)]
    return df.head(config["max_samples"]).reset_index(drop=True)

def load_audio(file_path, target_sr):
    try:
        audio, _ = librosa.load(file_path, sr=target_sr, mono=True)
        return audio.astype(np.float32)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None

def run_inference():
    logger.info(f"Loading model: {CONFIG['model_name']}")
    processor = AutoProcessor.from_pretrained(CONFIG["model_name"])
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    ).to(DEVICE)

    df = load_metadata(CONFIG)
    predictions = []

    logger.info(f"Starting inference on {len(df)} samples...")
    for i in tqdm(range(len(df))):
        audio = load_audio(df.iloc[i]["audio_path"], CONFIG["target_sampling_rate"])
        if audio is None:
            predictions.append("")
            continue
            
        inputs = processor(audio, sampling_rate=CONFIG["target_sampling_rate"], return_tensors="pt").to(DEVICE)
        input_features = inputs.input_features.to(model.dtype)

        with torch.no_grad():
            generated_ids = model.generate(input_features, language="azerbaijani", task="transcribe")
        
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        predictions.append(transcription.strip())

    df["prediction"] = predictions
    
    # Compute metrics
    df["wer_sample"] = [jiwer.wer(normalize_text(r), normalize_text(h)) for r, h in zip(df["sentence"], df["prediction"])]
    df["cer_sample"] = [jiwer.cer(normalize_text(r), normalize_text(h)) for r, h in zip(df["sentence"], df["prediction"])]

    corpus_wer = jiwer.wer([normalize_text(r) for r in df["sentence"]], [normalize_text(h) for h in df["prediction"]])
    corpus_cer = jiwer.cer([normalize_text(r) for r in df["sentence"]], [normalize_text(h) for h in df["prediction"]])

    logger.info(f"Corpus WER: {corpus_wer*100:.2f}% | CER: {corpus_cer*100:.2f}%")

    # Save results
    output_path = "../results/part_a_inference_results.csv"
    os.makedirs("../results", exist_ok=True)
    df[["sentence", "prediction", "wer_sample", "cer_sample"]].to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    run_inference()
