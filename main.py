# pre_train_translation_model.py
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from bs4 import BeautifulSoup
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import pandas as pd
import html
import ebooklib
from ebooklib import epub# pre_train_translation_model.py

from huggingface_hub import login
login(token="")

# 1. Konfigurasi
class Config:
    EPUB_PATH = "معالم السنة النبوية -.epub"  # Ganti dengan path EPUB Anda
    MODEL_NAME = "Helsinki-NLP/opus-mt-ar-en"
    SAVE_DIR = "fine_tuned_model"
    TARGET_LANG = "id"
    BATCH_SIZE = 4
    EPOCHS = 3
    LEARNING_RATE = 3e-5
    MAX_LENGTH = 256
    MIN_SENTENCE_LENGTH = 15
    MAX_SENTENCE_LENGTH = 300

# 2. Ekstraksi EPUB yang Diperbaiki
class EPUBProcessor:
    def __init__(self):
        self.ns = {
            'container': 'urn:oasis:names:tc:opendocument:xmlns:container',
            'opf': 'http://www.idpf.org/2007/opf',
            'dc': 'http://purl.org/dc/elements/1.1/'
        }

    def _extract_text_from_element(self, element):
        text = []
        if element.text:
            text.append(element.text.strip())
        for child in element:
            text.extend(self._extract_text_from_element(child))
            if child.tail:
                text.append(child.tail.strip())
        return text

    def process_epub(self, file_path):
        book = epub.read_epub(file_path)
        texts = []

        # Ekstrak konten utama
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')

                # Bersihkan tag yang tidak diperlukan
                for tag in soup(['script', 'style', 'meta', 'link']):
                    tag.decompose()

                # Ekstrak teks dari berbagai elemen
                text_elements = soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'li'])
                for el in text_elements:
                    text = el.get_text(separator=' ', strip=True)
                    if text:
                        texts.append(html.unescape(text))

        return ' '.join(texts)

# 3. Preprocessing Teks
class TextPreprocessor:
    def __init__(self):
        self.sentence_tokenizer = re.compile(
            r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\。|\？|\！)\s+'
        )

    def clean_text(self, text):
        # Normalisasi karakter khusus
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        return text.strip()

    def split_sentences(self, text):
        sentences = []
        for chunk in re.split(r'\n+', text):
            chunk = chunk.strip()
            if chunk:
                sentences.extend([s.strip() for s in self.sentence_tokenizer.split(chunk) if s.strip()])
        return sentences

    def filter_sentences(self, sentences):
        return [
            s for s in sentences
            if Config.MIN_SENTENCE_LENGTH <= len(s) <= Config.MAX_SENTENCE_LENGTH
        ]

# 4. Pipeline Training
class TranslationTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, use_auth_token=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_NAME, use_auth_token=True).to(self.device)

    def create_dataset(self, sentences):
        translations = []
        for batch in self._chunked(sentences, Config.BATCH_SIZE):
            inputs = self.tokenizer(
                batch,
                max_length=Config.MAX_LENGTH,
                truncation=True,
                padding='longest',
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model.generate(**inputs)
            translated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translations.extend(translated)

        return pd.DataFrame({
            'source': sentences,
            'target': translations
        })

    def _chunked(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def train(self, train_df):
        dataset = Dataset.from_pandas(train_df)
        dataset = dataset.train_test_split(test_size=0.1)

        def preprocess(examples):
            inputs = [ex for ex in examples['source']]
            targets = [ex for ex in examples['target']]

            model_inputs = self.tokenizer(
                inputs,
                max_length=Config.MAX_LENGTH,
                truncation=True,
                padding='max_length'
            )

            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    targets,
                    max_length=Config.MAX_LENGTH,
                    truncation=True,
                    padding='max_length'
                )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized_ds = dataset.map(
            preprocess,
            batched=True,
            remove_columns=dataset["train"].column_names
        )

        args = Seq2SeqTrainingArguments(
            output_dir=Config.SAVE_DIR,
            evaluation_strategy="epoch",
            learning_rate=Config.LEARNING_RATE,
            per_device_train_batch_size=Config.BATCH_SIZE,
            per_device_eval_batch_size=Config.BATCH_SIZE,
            num_train_epochs=Config.EPOCHS,
            weight_decay=0.01,
            save_total_limit=3,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            logging_steps=100,
            report_to="none"
        )

        trainer = Seq2SeqTrainer(
            self.model,
            args,
            train_dataset=tokenized_ds["train"],
            eval_dataset=tokenized_ds["test"],
            data_collator=DataCollatorForSeq2Seq(self.tokenizer),
            tokenizer=self.tokenizer
        )

        print("Memulai training...")
        trainer.train()
        self.model.save_pretrained(Config.SAVE_DIR)
        self.tokenizer.save_pretrained(Config.SAVE_DIR)

# 5. Pipeline Utama
def main():
    # Ekstraksi teks
    print("Memproses EPUB...")
    epub_processor = EPUBProcessor()
    raw_text = epub_processor.process_epub(Config.EPUB_PATH)

    # Preprocessing
    print("Membersihkan teks...")
    preprocessor = TextPreprocessor()
    clean_text = preprocessor.clean_text(raw_text)
    sentences = preprocessor.split_sentences(clean_text)
    filtered = preprocessor.filter_sentences(sentences)

    print(f"Ditemukan {len(filtered)} kalimat valid")

    if len(filtered) == 0:
        print("Error: Tidak ada teks yang berhasil diekstraksi!")
        print("Penyebab mungkin:")
        print("- Format EPUB tidak standar")
        print("- Dokumen terproteksi/terenkripsi")
        print("- Struktur konten tidak terdeteksi")
        return

    # Training
    print("Mempersiapkan training...")
    trainer = TranslationTrainer()
    train_df = trainer.create_dataset(filtered)

    print("Memulai training model...")
    trainer.train(train_df)
    print(f"Model disimpan di: {Config.SAVE_DIR}")

if __name__ == "__main__":
    main()
