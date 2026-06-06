# Movie Content-Based Recommender System

Aplikasi ini adalah content-based recommender system yang merekomendasikan film berdasarkan metadata film seperti title, genre, dan tag.

## Dataset

Dataset: MovieLens Latest Small

File yang digunakan:
- ratings.csv
- movies.csv
- tags.csv

## Method

Sistem menggunakan sentence embedding dari `sentence-transformers/all-MiniLM-L6-v2`.

Tahapan:
1. Menggabungkan title, genre, dan tag menjadi item text.
2. Mengubah item text menjadi embedding.
3. Membuat user profile embedding dari film yang disukai user.
4. Menghitung cosine similarity antara user profile dan semua film.
5. Mengambil top-N film yang belum pernah dirating oleh user.

## Evaluation

Metrik:
- Precision@10
- Recall@10
- NDCG@10

## How to Run

Install dependency:

```bash
pip install -r requirements.txt

## Kelompok 

- Abdullah (103012330146)
- Muhammad Febrian Hafiz 
- Hilmi Musyafa 
- Arief Bagas Nugraha