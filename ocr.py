import pytesseract
from PIL import Image

def baca_teks_dari_gambar(nama_file_gambar):
    try:
        gambar = Image.open(nama_file_gambar)
        teks = pytesseract.image_to_string(gambar, lang='eng')
        return teks
    except Exception as e:
        return f"Terjadi kesalahan saat membaca gambar: {e}"
