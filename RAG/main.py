# main.py
import os
import uuid
import hashlib
import logging
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from enum import Enum
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from translator import InteractiveTranslator, setup_logging

# --- KONFIGURASI DAN STATE GLOBAL ---

# Mengatur logging
setup_logging('api_translation.log')

# Variabel global untuk menyimpan instance translator dan cache
# Ini penting agar model hanya dimuat sekali saat startup.
state = {}

# Membuat direktori yang diperlukan jika belum ada
os.makedirs("temp", exist_ok=True)
os.makedirs("cache", exist_ok=True)

# --- LIFESPAN MANAGER UNTUK MEMUAT MODEL SAAT STARTUP ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Fungsi yang dieksekusi saat startup dan shutdown.
    Model AI yang berat akan dimuat di sini.
    """
    logging.info("Server startup: Memulai proses pemuatan model...")
    
    # Konfigurasi model (bisa dipindahkan ke file config jika perlu)
    model_id = "Qwen/Qwen2-1.5B-Instruct"
    
    # Inisialisasi translator dan simpan di state global
    state['translator'] = InteractiveTranslator(model_id=model_id, cache_dir="cache")
    state['translator'].load_model() # Memuat model dan tokenizer
    
    # Cache untuk menyimpan chunk yang sudah dipindai dari file
    # Key: hash file, Value: list of chunks
    state['chunk_cache'] = {}
    
    # Cache untuk menyimpan path file sementara
    # Key: hash file, Value: path file
    state['file_path_cache'] = {}
    
    logging.info("Model berhasil dimuat. Server siap menerima permintaan.")
    
    yield # Aplikasi berjalan di sini
    
    # Kode setelah yield akan dieksekusi saat shutdown
    logging.info("Server shutdown.")
    state.clear()


# --- INISIALISASI APLIKASI FASTAPI ---

app = FastAPI(
    title="API Penerjemah EPUB",
    description="API untuk memproses dan menerjemahkan konten dari file EPUB chunk per chunk.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enum untuk bahasa target agar input lebih terstruktur
class TargetLanguage(str, Enum):
    english = "English"
    indonesian = "Indonesian"
    malay = "Malay"
    japanese = "Japanese"
    korean = "Korean"

# --- ENDPOINTS API ---

@app.post("/total-chunk", summary="Menganalisis EPUB dan Mendapatkan Jumlah Chunk")
async def get_total_chunks(file: UploadFile = File(..., description="File EPUB yang akan dianalisis.")):
    """
    Endpoint ini menerima file EPUB, memindainya untuk mengekstrak semua kalimat
    unik (chunk), dan mengembalikan jumlah total chunk yang ditemukan.

    Proses ini di-cache berdasarkan konten file. Jika file yang sama diunggah lagi,
    hasil akan dikembalikan dari cache tanpa memindai ulang.
    """
    try:
        contents = await file.read()
        file_hash = hashlib.sha256(contents).hexdigest()

        # Cek apakah hasil scan untuk file ini sudah ada di cache
        if file_hash in state['chunk_cache']:
            logging.info(f"Cache hit untuk file hash: {file_hash[:10]}...")
            total_chunks = len(state['chunk_cache'][file_hash])
            return {"total": total_chunks, "file_id": file_hash}

        logging.info(f"Cache miss. Memproses file baru dengan hash: {file_hash[:10]}...")
        
        # Simpan file ke direktori sementara
        temp_filename = f"{uuid.uuid4()}.epub"
        temp_filepath = os.path.join("temp", temp_filename)
        with open(temp_filepath, "wb") as f:
            f.write(contents)

        # Pindai file untuk mendapatkan semua chunk
        all_chunks = state['translator'].scan_and_get_chunks(epub_path=temp_filepath)
        
        # Simpan hasil scan dan path file ke cache
        state['chunk_cache'][file_hash] = all_chunks
        state['file_path_cache'][file_hash] = temp_filepath

        logging.info(f"File berhasil dipindai. Ditemukan {len(all_chunks)} chunk.")
        
        return {"total": len(all_chunks), "file_id": file_hash}

    except Exception as e:
        logging.error(f"Error di /total-chunk: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/process-chunk", summary="Menerjemahkan Satu Chunk Spesifik")
async def process_chunk(
    file_id: str = Form(..., description="ID unik file yang didapat dari endpoint /total-chunk."),
    chunk: int = Form(..., gt=0, description="Nomor chunk yang akan diterjemahkan (dimulai dari 1)."),
    target_language: TargetLanguage = Form(TargetLanguage.indonesian, description="Bahasa target terjemahan.")
):
    """
    Endpoint ini menerjemahkan satu chunk (kalimat) dari file yang sudah diproses sebelumnya.
    Anda harus memanggil `/total-chunk` terlebih dahulu untuk mendapatkan `file_id`.
    """
    try:
        # Validasi file_id
        if file_id not in state['chunk_cache']:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File ID tidak ditemukan. Harap unggah file melalui /total-chunk terlebih dahulu."
            )

        all_chunks = state['chunk_cache'][file_id]
        total_chunks = len(all_chunks)

        # Validasi nomor chunk
        if not (1 <= chunk <= total_chunks):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Nomor chunk tidak valid. Harap masukkan angka antara 1 dan {total_chunks}."
            )
        
        chunk_to_translate = all_chunks[chunk - 1]
        
        logging.info(f"Menerjemahkan chunk #{chunk} dari file {file_id[:10]}... ke {target_language.value}")
        
        # Panggil fungsi terjemahan dari instance global
        translated_text = state['translator'].get_single_translation(
            chunk_to_translate=chunk_to_translate,
            target_language=target_language.value,
            book_hash=file_id # Menggunakan file_id sebagai ID unik untuk cache terjemahan
        )
        
        return {"output": translated_text, "original": chunk_to_translate, "chunk_number": chunk}

    except Exception as e:
        logging.error(f"Error di /process-chunk: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Selamat datang di API Penerjemah EPUB. Kunjungi /docs untuk dokumentasi."}
