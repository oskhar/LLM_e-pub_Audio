# -*- coding: utf-8 -*-
import os
import torch
import gc
import json
import argparse
import logging
import uuid
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import ebooklib # solusi: mengimpor seluruh library ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString
from tqdm import tqdm
from huggingface_hub import HfFolder

# --- KONFIGURASI DAN SETUP LOGGING ---
def setup_logging(log_file='translation_interactive.log'):
    """Mengatur logging untuk menyimpan output ke file dan menampilkan di konsol."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def sentence_splitter(text):
    """Memecah teks menjadi kalimat. Dibuat sederhana untuk kompatibilitas."""
    # Regex untuk memecah berdasarkan tanda baca akhir, termasuk tanda baca Arab.
    sentences = re.split(r'(?<=[.!?؟۔])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# --- KELAS UTAMA UNTUK PENERJEMAHAN ---
class InteractiveTranslator:
    """
    Kelas profesional untuk menerjemahkan EPUB secara interaktif, kalimat per kalimat.
    Fitur: Pemuatan model on-demand, Kuantisasi 4-bit, Caching Terjemahan, Output Teks.
    """
    def __init__(self, config):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        # Path untuk cache terjemahan berdasarkan nama file input
        base, _ = os.path.splitext(self.config.input_path)
        self.cache_path = f"{base}.translation_cache.json"

    def load_model(self):
        """Memuat model dan tokenizer. Dipanggil secara eksplisit saat terjemahan pertama diminta."""
        if self.model is not None:
            return # Model sudah dimuat
            
        logging.info(f"Loading model for the first time: {self.config.model_id}...")
        if self.device == "cpu":
            logging.warning("Running on CPU. This will be extremely slow. A GPU is highly recommended.")
        
        token = HfFolder.get_token()
        if not token:
            logging.warning("Hugging Face token not found. Downloads may be slower.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id, token=token, trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=token
        )
        logging.info("Model loaded successfully. Ready to translate.")

    def scan_and_get_chunks(self):
        """Memindai EPUB dan mengekstrak semua kalimat unik sebagai 'chunk'."""
        if not os.path.exists(self.config.input_path):
            logging.error(f"Error: EPUB file not found at '{self.config.input_path}'")
            return []

        logging.info(f"Scanning EPUB file: {self.config.input_path} to find all translatable sentences...")
        # SOLUSI: Menggunakan ebooklib.epub.read_epub
        book = epub.read_epub(self.config.input_path)
        all_sentences = set()

        # SOLUSI: Menggunakan ebooklib.ITEM_DOCUMENT
        for item in tqdm(book.get_items_of_type(ebooklib.ITEM_DOCUMENT), desc="Scanning items"):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text_nodes = soup.find_all(string=True)
            for text_node in text_nodes:
                if isinstance(text_node, NavigableString) and text_node.strip():
                    sentences_from_node = sentence_splitter(text_node.strip())
                    for sentence in sentences_from_node:
                        # Filter untuk kalimat yang signifikan (lebih dari 2 kata)
                        if len(sentence.split()) > 2:
                            all_sentences.add(sentence)
        
        # Urutkan untuk memastikan urutan chunk konsisten setiap kali program dijalankan
        return sorted(list(all_sentences))

    def get_single_translation(self, chunk_to_translate):
        """Menerjemahkan satu chunk, menggunakan cache jika tersedia."""
        translation_cache = self._load_cache()
        
        if chunk_to_translate in translation_cache:
            logging.info(f"Translation found in cache.")
            return translation_cache[chunk_to_translate]

        logging.info(f"Translating new chunk...")
        
        # Buat pesan sesuai dengan format chat Qwen2
        messages = [
            {"role": "system", "content": "You are an expert translator."},
            {"role": "user", "content": f"Translate the following Arabic text to {self.config.target_language}. Provide only the translation, without any additional text or explanations.\n\nArabic text: \"{chunk_to_translate}\""}
        ]
        
        # Gunakan apply_chat_template untuk memformat prompt
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(inputs.input_ids, max_new_tokens=1024, do_sample=False)
        
        # Decode hanya bagian yang baru digenerate
        translation = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
        
        # Simpan hasil terjemahan baru ke cache
        translation_cache[chunk_to_translate] = translation
        self._save_cache(translation_cache)
        
        return translation

    def _load_cache(self):
        """Memuat cache terjemahan dari file JSON."""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logging.warning(f"Could not read cache file: {e}. Starting with an empty cache.")
        return {}

    def _save_cache(self, state):
        """Menyimpan cache terjemahan ke file JSON."""
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=4)
        except IOError as e:
            logging.error(f"Could not save cache file: {e}")

def main():
    """Fungsi utama untuk menjalankan alur kerja interaktif."""
    parser = argparse.ArgumentParser(
        description="Interactively translate a single sentence (chunk) from an Arabic EPUB.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_path", help="Path to the source Arabic EPUB file.")
    parser.add_argument("-lang", "--target_language", default="English", help="Target language for translation (e.g., 'English', 'Indonesian').")
    parser.add_argument("--model_id", default="Qwen/Qwen2-1.5B-Instruct", help="Model ID from Hugging Face.")
    parser.add_argument("--log_file", default="translation_interactive.log", help="File to store logs.")
    
    args = parser.parse_args()

    setup_logging(args.log_file)

    try:
        translator = InteractiveTranslator(args)
        
        # Fase 1: Analisis
        all_chunks = translator.scan_and_get_chunks()
        total_chunks = len(all_chunks)

        if not all_chunks:
            logging.error("No translatable sentences found in the EPUB.")
            return

        print("\n===================================================================")
        print(f" Analisis Selesai. Ditemukan total {total_chunks} kalimat (chunk) unik.")
        print("===================================================================\n")

        # Fase 2: Interaktif
        model_loaded = False
        while True:
            user_input = input(f"Masukkan nomor chunk untuk diterjemahkan (1-{total_chunks}), atau 'q' untuk keluar: ").strip().lower()
            
            if user_input == 'q':
                print("Terima kasih! Program berhenti.")
                break
            
            try:
                chunk_num = int(user_input)
                if not (1 <= chunk_num <= total_chunks):
                    raise ValueError
            except ValueError:
                print(f"Input tidak valid. Harap masukkan angka antara 1 dan {total_chunks}.")
                continue

            # Memuat model hanya saat diperlukan
            if not model_loaded:
                translator.load_model()
                model_loaded = True

            # Dapatkan chunk yang akan diterjemahkan
            chunk_to_process = all_chunks[chunk_num - 1]
            
            # Terjemahkan satu chunk
            translated_text = translator.get_single_translation(chunk_to_process)
            
            # Tampilkan hasil
            print("\n------------------ HASIL TERJEMAHAN ------------------")
            print(f"Chunk #{chunk_num}:\n")
            print(f"  ASLI (Arab)    : {chunk_to_process}")
            print(f"  TERJEMAHAN     : {translated_text}")
            print("------------------------------------------------------\n")

    except Exception as e:
        logging.error("An unexpected error occurred in the main pipeline.", exc_info=True)

if __name__ == "__main__":
    main()
