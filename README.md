---
title: COMS AI Service
emoji: 🍱
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# COMS AI Service

Layanan estimasi kepadatan kantin untuk sistem **C.O.M.S (Canteen Occupancy Monitoring System)**. Dibangun menggunakan FastAPI dan model deep learning CSRNet berbasis PyTorch yang mampu menghitung jumlah kepala (head count) dari gambar kamera kantin secara otomatis.

## Cara Kerja

1. Gambar dikirim ke endpoint `/api/predict` melalui form multipart
2. Model CSRNet memproses gambar dan menghasilkan estimasi jumlah orang
3. Hasil prediksi diteruskan secara otomatis ke backend utama

## Requirements

- Python 3.10+
- pip

## Instalasi & Menjalankan Lokal

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 7860
```

Server berjalan di `http://localhost:7860`.

## Environment Variables

| Variable | Keterangan |
|----------|------------|
| `BACKEND_URL` | URL endpoint `POST /api/predictions` pada backend utama |
| `MODEL_WEIGHTS_PATH` | (Opsional) Path ke file weights CSRNet `.pth` |

Buat file `.env` di root project dan isi sesuai tabel di atas.

## Weights Model

Letakkan file checkpoint CSRNet (contoh: dilatih pada ShanghaiTech Part A) ke path `weights/csrnet.pth`. Model akan dimuat otomatis saat `MODEL_WEIGHTS_PATH` di-set atau file default tersebut ditemukan.

## Endpoints

| Method | Path | Keterangan |
|--------|------|------------|
| GET | `/` | Health check |
| POST | `/api/predict` | Kirim gambar (`file`) dan `canteen_id`, mendapat hasil head count |

## Deploy

Layanan ini di-deploy ke **Hugging Face Spaces** menggunakan Docker.
