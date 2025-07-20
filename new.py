import os
import re
import torch
from ebooklib import epub
from ebooklib.utils import debug
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM

# === KONFIGURASI ===
CHECKPOINT = "bigscience/bloomz-7b1-mt"
CACHE_DIR = "./model_cache/bloomz-7b1-mt"
EPUB_PATH = 'تيسير اللطيف المنان في خلاصة تفسير القرآن - ط الأوقاف السعودية.epub'
TARGET_LANGUAGE = "English"
MAX_INPUT_LENGTH = 256  # Diperkecil agar lebih ringan diproses
MAX_OUTPUT_LENGTH = 512  # Dikurangi untuk mempercepat output


def extract_text_from_epub(epub_path):
    from ebooklib import ITEM_DOCUMENT
    book = epub.read_epub(epub_path)
    paragraphs = []

    for item in list(book.get_items()):
        if item.get_type() == ITEM_DOCUMENT:
            try:
                content = item.get_content().decode('utf-8', errors='ignore')
            except Exception:
                continue

            soup = BeautifulSoup(content, 'html.parser')
            for tag in soup(['script', 'style']):
                tag.decompose()
            text = soup.get_text(separator=' ', strip=True)

            print(f"\n--- [DEBUG] File: {item.get_name()} ---")
            print(text[:500])

            matches = re.findall(r'([\u0600-\u06FF\s\d\W]{40,})', text)
            for match in matches:
                cleaned = re.sub(r'\s+', ' ', match).strip()
                paragraphs.append(cleaned)

    return paragraphs


def split_paragraphs(paragraphs, max_chunk_length=MAX_INPUT_LENGTH):
    chunks, current = [], ''
    for para in paragraphs:
        if len(current) + len(para) < max_chunk_length:
            current += para + ' '
        else:
            chunks.append(current.strip())
            current = para + ' '
    if current:
        chunks.append(current.strip())
    return chunks


def load_model(checkpoint=CHECKPOINT, cache_dir=CACHE_DIR):
    print(f"Memuat model dari {cache_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        cache_dir=cache_dir,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map={"": "cpu"}
    )
    return tokenizer, model


def translate_text(text, tokenizer, model, target_language=TARGET_LANGUAGE):
    prompt = (
        f"Translate the following Islamic Arabic text to {target_language}.\n"
        f"Text: {text}\n\nTranslation:"
    )
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=MAX_INPUT_LENGTH).to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=MAX_OUTPUT_LENGTH,
        num_beams=2,  # Kurangi beam untuk kecepatan
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    print("\U0001F4D6 Mengekstrak teks dari EPUB...")
    paragraphs = extract_text_from_epub(EPUB_PATH)
    print(f"✅ Ditemukan {len(paragraphs)} paragraf yang valid.")

    chunks = split_paragraphs(paragraphs)
    print(f"\U0001F4E6 Total chunk siap terjemah: {len(chunks)}")

    if not chunks:
        print("\n❌ Tidak ada teks yang berhasil diambil. Coba periksa isi EPUB atau tambahkan debug print.")
        return

    for i, chunk in enumerate(chunks):
        print(f"[Chunk {i}] {chunk[:100]}...\n")

    try:
        selected_chunk = int(input(f"Pilih nomor chunk (0 sampai {len(chunks)-1}) yang ingin diterjemahkan: "))
    except ValueError:
        print("Input tidak valid. Menggunakan chunk 0.")
        selected_chunk = 0

    if selected_chunk < 0 or selected_chunk >= len(chunks):
        print("Nomor chunk di luar rentang. Menggunakan chunk 0.")
        selected_chunk = 0

    tokenizer, model = load_model()

    print("\n=== HASIL TERJEMAHAN ===\n")
    print(f"\n--- Chunk {selected_chunk} ---")
    translated = translate_text(chunks[selected_chunk], tokenizer, model)
    print(translated)


if __name__ == "__main__":
    main()
