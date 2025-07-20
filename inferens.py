# inference_translation_model.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from main import Config, TextPreprocessor  # Mengimpor konfigurasi dan preprocessor yang sama
import argparse
import time

class TranslationInference:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.preprocessor = TextPreprocessor()
        self.model.eval()  # Mode evaluasi

    def preprocess_input(self, text):
        """Proses teks input sama seperti saat training"""
        clean_text = self.preprocessor.clean_text(text)
        sentences = self.preprocessor.split_sentences(clean_text)
        return self.preprocessor.filter_sentences(sentences)

    def translate(self, text, batch_size=4, max_length=256):
        # Preprocess teks input
        sentences = self.preprocess_input(text)
        print(f"Memproses {len(sentences)} kalimat...")

        translations = []

        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]

                # Tokenisasi batch
                inputs = self.tokenizer(
                    batch,
                    max_length=max_length,
                    truncation=True,
                    padding="longest",
                    return_tensors="pt"
                ).to(self.device)

                # Generate terjemahan
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=max_length,
                    num_beams=5,  # Meningkatkan kualitas terjemahan
                    early_stopping=True
                )

                # Decode hasil
                batch_translations = self.tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )

                translations.extend(batch_translations)

        return translations

    def format_translation(self, source, translation):
        """Format hasil terjemahan dengan alignment"""
        max_lengths = [max(len(s), len(t)) for s, t in zip(source, translation)]
        max_length = max(max_lengths) if max_lengths else 0
        border = "+" + "-"*(max_length + 2) + "+"
        return "\n".join(
            f"| {s.ljust(max_length)} |\n| {t.ljust(max_length)} |\n{border}"
            for s, t in zip(source, translation)
        )

def main():
    parser = argparse.ArgumentParser(description="Terjemahkan teks menggunakan model terlatih")
    parser.add_argument("--text", type=str, help="Teks langsung untuk diterjemahkan")
    parser.add_argument("--input-file", type=str, help="Path ke file teks input")
    parser.add_argument("--output-file", type=str, help="Path ke file output")
    parser.add_argument("--batch-size", type=int, default=4, help="Ukuran batch untuk inference")
    args = parser.parse_args()

    # Inisialisasi sistem terjemahan
    translator = TranslationInference(Config.SAVE_DIR)

    # Baca input
    if args.text:
        input_text = args.text
    elif args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            input_text = f.read()
    else:
        print("Harap berikan input teks atau file!")
        return

    # Lakukan terjemahan
    start_time = time.time()
    sentences = translator.preprocess_input(input_text)
    translations = translator.translate(input_text, batch_size=args.batch_size)
    end_time = time.time()

    # Format output
    formatted = translator.format_translation(sentences, translations)

    # Tampilkan atau simpan hasil
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(formatted)
        print(f"Hasil terjemahan disimpan di: {args.output_file}")
    else:
        print("\nHasil Terjemahan:")
        print(formatted)

    print(f"\nWaktu total: {end_time - start_time:.2f} detik")
    print(f"Jumlah kalimat: {len(sentences)}")
    print(f"Kecepatan: {len(sentences)/(end_time - start_time):.2f} kalimat/detik")

if __name__ == "__main__":
    main()
