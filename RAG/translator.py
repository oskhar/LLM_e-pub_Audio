# translator.py
import os
import torch
import gc
import json
import logging
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString
from tqdm import tqdm
from huggingface_hub import HfFolder

# --- FUNGSI UTILITAS ---
def setup_logging(log_file='translation_api.log'):
    """Mengatur logging untuk menyimpan output ke file dan menampilkan di konsol."""
    # Mencegah duplikasi handler jika fungsi ini dipanggil berkali-kali
    if logging.getLogger().hasHandlers():
        logging.getLogger().handlers.clear()
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def sentence_splitter(text):
    """Memecah teks menjadi kalimat. Dibuat sederhana untuk kompatibilitas."""
    sentences = re.split(r'(?<=[.!?؟۔])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# --- KELAS UTAMA UNTUK PENERJEMAHAN ---
class InteractiveTranslator:
    """
    Kelas profesional untuk menerjemahkan. Didesain untuk digunakan dalam API.
    Model dimuat sekali, dan fungsi-fungsi lain beroperasi berdasarkan permintaan.
    """
    def __init__(self, model_id, cache_dir="cache"):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        os.makedirs(self.cache_dir, exist_ok=True)

    def load_model(self):
        """Memuat model dan tokenizer. Dipanggil sekali saat server startup."""
        if self.model is not None:
            return
            
        logging.info(f"Memuat model untuk pertama kali: {self.model_id}...")
        if self.device == "cpu":
            logging.warning("Berjalan di CPU. Proses ini akan sangat lambat. GPU sangat direkomendasikan.")
        
        token = HfFolder.get_token()
        if not token:
            logging.warning("Token Hugging Face tidak ditemukan. Proses unduh mungkin lebih lambat.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, token=token, trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            token=token
        )
        logging.info("Model berhasil dimuat dan siap digunakan.")

    def scan_and_get_chunks(self, epub_path):
        """Memindai EPUB dari path yang diberikan dan mengekstrak semua kalimat unik."""
        if not os.path.exists(epub_path):
            logging.error(f"Error: File EPUB tidak ditemukan di '{epub_path}'")
            return []

        logging.info(f"Memindai file EPUB: {os.path.basename(epub_path)} untuk menemukan semua kalimat...")
        book = epub.read_epub(epub_path)
        all_sentences = set()

        for item in tqdm(book.get_items_of_type(ebooklib.ITEM_DOCUMENT), desc="Memindai item"):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text_nodes = soup.find_all(string=True)
            for text_node in text_nodes:
                if isinstance(text_node, NavigableString) and text_node.strip():
                    sentences_from_node = sentence_splitter(text_node.strip())
                    for sentence in sentences_from_node:
                        # Filter untuk kalimat yang signifikan (lebih dari 2 kata)
                        if len(sentence.split()) > 2:
                            all_sentences.add(sentence)
        
        return sorted(list(all_sentences))

    def get_single_translation(self, chunk_to_translate, target_language, book_hash):
        """Menerjemahkan satu chunk, menggunakan cache spesifik untuk buku tersebut."""
        cache_path = os.path.join(self.cache_dir, f"{book_hash}.translation_cache.json")
        translation_cache = self._load_cache(cache_path)
        
        if chunk_to_translate in translation_cache:
            logging.info(f"Terjemahan ditemukan di cache untuk chunk: '{chunk_to_translate[:30]}...'")
            return translation_cache[chunk_to_translate]

        logging.info(f"Menerjemahkan chunk baru: '{chunk_to_translate[:30]}...'")
        
        messages = [
            {"role": "system", "content": "You are an expert translator."},
            {"role": "user", "content": f"Translate the following Arabic text to {target_language}. Provide only the translation, without any additional text or explanations.\n\nArabic text: \"{chunk_to_translate}\""}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids, 
                max_new_tokens=1024, 
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id # Mencegah warning
            )
        
        translation = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
        
        translation_cache[chunk_to_translate] = translation
        self._save_cache(translation_cache, cache_path)
        
        # Membersihkan memori GPU setelah setiap generasi
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            
        return translation

    def _load_cache(self, path):
        """Memuat cache terjemahan dari file JSON."""
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logging.warning(f"Tidak dapat membaca file cache: {e}. Memulai dengan cache kosong.")
        return {}

    def _save_cache(self, state, path):
        """Menyimpan cache terjemahan ke file JSON."""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=4)
        except IOError as e:
            logging.error(f"Tidak dapat menyimpan file cache: {e}")
