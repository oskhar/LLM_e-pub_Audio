import os
import re
import torch
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ebooklib import epub
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM

# === KONFIGURASI ===
CHECKPOINT = "bigscience/bloomz-7b1-mt"
CACHE_DIR = "./model_cache/bloomz-7b1-mt"
MAX_INPUT_LENGTH = 256
MAX_OUTPUT_LENGTH = 512

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = None
model = None
chunks = []


def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, cache_dir=CACHE_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT,
            cache_dir=CACHE_DIR,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map={"": "cpu"}
        )


def extract_text_from_epub(epub_file):
    from ebooklib import ITEM_DOCUMENT
    book = epub.read_epub(epub_file)
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


def translate_text(text, target_language):
    load_model()
    prompt = f"Translate the following Islamic Arabic text to {target_language}.\nText: {text}\n\nTranslation:"
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=MAX_INPUT_LENGTH).to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=MAX_OUTPUT_LENGTH,
        num_beams=2,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split('Translation:')[-1].strip()


@app.post("/total-chunk")
async def get_total_chunk(file: UploadFile = File(...)):
    global chunks
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name
    paragraphs = extract_text_from_epub(tmp_path)
    chunks = split_paragraphs(paragraphs)
    os.remove(tmp_path)
    return {"total": len(chunks)}


@app.post("/process-chunk")
async def process_chunk(file: UploadFile = File(...), chunk: int = Form(...), target_language: str = Form(...)):
    if chunk < 0 or chunk >= len(chunks):
        return JSONResponse(status_code=400, content={"error": "Invalid chunk index."})
    text_chunk = chunks[chunk]
    result = translate_text(text_chunk, target_language)
    return {"output": result}
