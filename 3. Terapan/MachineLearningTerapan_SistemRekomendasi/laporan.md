# Laporan Proyek Pertama

### Aldo Valentino | M180X0298

## Project Overview

Sistem rekomendasi movie merupakan sistem yang merekomendasikan movie kepada penonton atau pengguna lainnya, sistem rekomendasi ini banyak diterapkan pada situs seperti netflix, iqiyi, wetv, dan lainnya. Sistem rekomendasi yang dibuat ini didasarkan dengan peferensi kesukaan atau tontonan terakhir pengguna serta rating dari movie tersebut.

Sistem rekomendasi telah menjadi lazim dalam beberapa tahun terakhir karena mereka menangani masalah kelebihan informasi dengan menyarankan pengguna produk yang paling relevan dari sejumlah besar data. Sistem rekomendasi membantu pengguna mengakses film pilihan mereka dengan menangkap _genre_ atau _tags_ yang persis sama di antara pengguna atau film dari riwayat atau penilaian pengguna lain. Namun, karena banyaknya tayangan yang bermacam - macam serta genre yang kian variatif diperlukan sistem rekomendasi yang tepat dengan akurasi yang cukup baik.

## Business Understanding

### Problem Statements

- Bagaimana cara merekomendasikan film atau tayangan yang disukai pengguna dapat direkomendasikan kepada pengguna lainnya ?

### Goals

- Membuat sistem rekomendasi dengan tingkat akurasi tinggi berdasarkan _rating_ dan aktivitas terakhir pengguna.

### Solution approach

Solusi yang dibuat yaitu dengan menggunakan 2 algoritma Machine Learning sistem rekomendasi, yaitu :

- **Content Based Filtering** adalah algoritma yang merekomendasikan item serupa dengan apa yang disukai pengguna, berdasarkan aktivitas mereka sebelumnya atau _feedback_ eksplisit.
- **Collaborative Filtering**. adalah algoritma yang bergantung pada pendapat atau _rating_ dari pengguna, tidak memerlukan atribut untuk setiap itemnya.

Algoritma Content Based Filtering digunakan untuk merekemondesikan movie berdasarkan aktivitas pengguna pada masa lalu, sedangkan algoritma Collabarative Filltering digunakan untuk merekomendasikan movie berdasarkan ratings yang paling tinggi.

## Data Understanding

Data atau dataset yang digunakan pada proyek machine learning ini adalah data **Movie Recommendation Data** yang didapat dari situs kaggle. Link dataset dapat dilihat dari tautan berikut [Dataset Sistem Rekomendasi Movie](https://www.kaggle.com/code/darpan25bajaj/movie-recommendation-system/data)

Variabel-variabel pada Dataset Sistem Rekomendasi Movie adalah sebagai berikut :

1. Links
   - movieID : ID tayangan pada tabel
   - imdbID : ID tayangan pada IMDB
   - tmdbID : ID dari _the movie database_
2. Movies
   - movieID : ID tayangan pada tabel
   - title : Judul tayangan
   - genre : ragam pada masing - masing tayangan, 1 tayangan bisa memiliki lebih dari 1 genre
3. Ratings
   - UserID : ID pengguna yang memberi penilaian
   - MovieID : ID tayangan yang diberi penilaian
   - Rating : nilai yang diberikan oleh pengguna
   - timestamp : Rekam waktu ketika pengguna memberikan penilaian
4. Tags
   - UserID : ID pengguna
   - MovieID : ID tayangan
   - tag : penanda pada tayangan
   - timestamp : rekam waktu

tahapan yang dilakukan mengenai data adalah dengan melakukan exploratory data analysis, yaitu dengan memperhatikan hubungan antar variabel bersarkan ID. Serta menggabungkan seluruh variabel movie_all berdasarkan movieId dan variabel user_all berdasarkan userId

## Data Preparation

Data preparation yang  digunakan sebagai berikut :

- Mengatasi missing value : menyeleksi data apakah data tersebut ada yang kosong atau tidak, jika ada data kosong maka data akan dihapus.
- Membagi data menjadi data training dan validasi
- Menggabungkan variabel : untuk menggabungkan beberapa variabel berdasarkan id yang sifatnya unik (berbeda dari yang lain).
  - movie_all : menggunakan _numpy.concatenate_ untuk menggabungkan nilai pada kolom movieID yang unik dari tabel links, movies, ratings, dan tags
  - user_all : menggunakan _numpy.concatenate_ untuk menggabungkan nilai pada kolom userID yang unik dari tabel ratings dan tags
  - all_movie_rate : menggabungkan dataframe movie dengan ratings, sehingga menjadi 1 tabel
  - all_movie_name : menggabungkan all_movie_rate  ddengan movies, movieID sebagai join
  - all_movie : menggabungkan all_movie_name dengan tags, movieID sebagai join
- Mengurutan data : untuk mengurutkan data berdasarkan movieId secara asceding.
- Mengatasi duplikasi data : untuk mengatasi data yang memiliki nilai atau isinya sama.
- Konversi data menjadi iist : untuk mengubah data menjadi daftar.
- Membuat dictionary : Untuk membuat dictionary dari data yang ada.
- Menggunakan TfidfVectorizer : untuk melakukan pembobotan.
- melakukan preprocessing : untuk menghilangkan permasalahan-permasalahan yang dapat mengganggu hasil daripada proses data
- mapping data : untuk memetakan data

## Modeling and Result

- Proses modeling yang  dilakukan pada data ini adalah dengan membuat algoritma machine learning, yaitu content based filtering dan collabrative filtering. untuk algoritma content based filtering dibuat dengan apa yang disukai pengguna pada masa lalu, sedangkan untuk content based filtering, dibuat dengan memanfaatkan tingkat rating dari movie tersebut.

- Berikut adalah hasil dari kedua algoritma tersebut :

  1. Hasil dari **content based filtering**  
      
      Cara kerja dari algoritma Content-Based Filtering adalah dengan mencari kesamaan pada riwayat aktivitas suatu pengguna kemudian memberi rekomendasi kepada pengguna lain yang memiliki riwayat aktivitas yang mirip, contohnya adalah genre pada suatu film.

      TFIDF bekerja dengan meningkatkan secara proporsional berapa kali sebuah kata muncul dalam dokumen tetapi diimbangi dengan jumlah dokumen yang menampilkannya. Oleh karena itu, kata-kata seperti 'ini', 'adalah', dan lainnya yang biasanya ada di semua dokumen tidak diberi peringkat yang sangat tinggi.

      berikut adalah movie yang disukai pengguna dimasa lalu :  
      |    Movie Name    |                      Genre                      |
      |:----------------:|:-----------------------------------------------:|
      | Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy |

     dari hasil di atas dapat dilihat bahwa pengguna menyukai movie yang berjudul Toy Story (1995) yang bergenre Adventure|Animation|Children|Comedy|Fantasy.Mendapatkan rekomendasi movie yang mirip dengan Toy Story (1995). Maka hasil top 5 rekomendasi berdasarkan algoritma conten based filtering adalah sebagai berikut :

      |                         Movie Name                        |                      Genre                      |
      |:---------------------------------------------------------:|:-----------------------------------------------:|
      |                                        Toy Story 2 (1999) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
      |                  Sinbad : Legend of the Seven Seas (2003) | Adventure\|Animation\|Children\|Fantasy         |
      | The Lord of the Rings : The Fellowship of the Ring (1978) | Adventure\|Animation\|Children\|Fantasy         |
      |                            Kiki's Delivery Service (1989) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
      |                                    The Cat Returns (2002) | Adventure\|Animation\|Children\|Fantasy         |

      dari hasil di atas dapat dilihat bahwa movie yang bergenre antar Adventure, Animation,Children, Comedy dan Fantasy menjadi yang direkomendasikan oleh sistem. Hal ini didasarkan pada kesukaan penonton atau pengguna pada masa lalu.

  2. hasil dari **collaborative filtering**  

      Cara kerja algoritma Collaborative Filtering didasarkan pada interaksi masa lalu suatu pengguna, dari data tersebut nantinya pengguna akan diklasifikasikan terhadap kelompok jenis yang serupa dan merekomendasikan setiap pengguna sesuai dengan preferensi kelompoknya. Sehingga akan menghasilkan rekomendasi yang baru.

      Cosine Similarity menghitung derajat kesamaan (similarity degree) antar movie dengan teknik cosine similarity.
      Menghitung kesamaan pengguna dengan  menemukan kesamaan antara dua pengguna (dapat menggunakan kesamaan kosinus). Sehingga  Cosine Similarity berarti kesamaan antara dua vektor ruang hasil kali dalam, diukur dengan kosinus sudut antara dua vektor.  Cosine Similarity juga dapat diterapkan pada Content-based Filtering
   
      berikut adalah tayangan berdasarkan penilaian:  

      - Penilaian tertinggi
          - |         Movie Name         |           Genre           |
            |:--------------------------:|:-------------------------:|
            |  Back to the Future (1985) | Adventure\|Comedy\|Sci-Fi |
            |        Going Places (1974) | Comedy\|Crime\|Drama      |
            |                 Saw (2004) | Horror\|MIstery\|Trhiller |
            | Love Me If You Dare (2003) | Drama\|Romance            |

      - Rekomendasi Film 10 teratas
          - |                Movie Name                |                Genre                |
            |:----------------------------------------:|:-----------------------------------:|
            |                    Paths of Glory (1957) | Drama\|War                          |
            |                 Last Day of Disco (1998) | Comedy\|Drama                       |
            |                              More (1998) | Animation\|Drama\|Sci-Fi            |
            |                    Midnight Clear (1992) | Drama\|War                          |
            |         Woman Under the Influence (1974) | Drama                               |
            |                        Adam's Rib (1949) | Comedy\|Romance                     |
            |                      Safety Last! (1923) | Action\|Comedy\|Romance             |
            |                          La Jetée (1962) | Romance\|Sci-Fi                     |
            |                    Into the Woods (1991) | Adnveture\|Comedy\|Fantasy\|Musical |
            | Reefer Madness: The Movie Musical (2005) | Comedy\|Drama\|Musical              |

      dari hasi di atas film yang memiliki _genre_ Comedy dan Drama menjadi movie yang paling tinggi penilaiannya. Kemudian top 10 movie yang direkomendasikan sistem adalah film dengan _genre_ Comedy, Drama, dan Romance.

## Evaluation

1. hasil Evaluasi untuk Content Based Filtering

      Teknik Evaluasi di atas adalah dengan menggunakan precission :

      ![rumusPrecission](img\rumusPrecission.png)

      Cara kerja dari formula di atas adalah dengan menentukan konten rekomendasi dari beberapa konten yang paling relevan, sehingga keluaran dari formula tersebut 2 konten paling relevan dan mirip dari 5 rekomendasi yang ada.

       Dari hasil rekomendasi di atas, diketahui bahwa Toy Story (1995) termasuk ke dalam genre Adventure|Animation|Children|Comedy|Fantasy. Dari 5 item yang direkomendasikan, 2 item memiliki genre Adventure|Animation|Children|Comedy|Fantasy (similar). Artinya, precision sistem tersebut sebesar 2/5 atau 40%. Dan 3/5 daintaranya memiliki 4 genre yang sama yakni Adventure|Animation|Children|Fantasy.

2. hasil Evaluasi untuk Collaborative Filtering

   Evaluasi metrik yang digunakan untuk mengukur kinerja model adalah metrik RMSE (Root Mean Squared Error).

   - RMSE adalah metode pengukuran dengan mengukur perbedaan nilai dari prediksi sebuah model sebagai estimasi atas nilai yang diobservasi. Root Mean Square Error adalah hasil dari akar kuadrat Mean Square Error. Keakuratan metode estimasi kesalahan pengukuran ditandai dengan adanya nilai RMSE yang kecil. Metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih kecil dikatakan lebih akurat daripada metode estimasi yang mempunyai Root Mean Square Error (RMSE) lebih besar

   - Kelebihan dan kekurang matriks ini adalah :

      - **kelebihan** : menekan kesalahan besar lebih sehingga bisa lebih tepat dalam beberapa kasus.  
      - **Kekurangan** : memberikan bobot yang relatif tinggi untuk kesalahan besar. Ini berarti RMSE harus lebih berguna ketika kesalahan besar sangat tidak diinginkan

   - formula dari matriks RMSE adalah sebagai berikut

   ![rumusRMSE](img\rumusRMSE.png)

   keterangan :
   - At : Nilai Aktual.
   - ft = Nilai hasil peramalan.
   - N = banyaknya dataset

   penerapan metrik tersebut adalah dengan menambahkan **_'metrics=[tf.keras.metrics.RootMeanSquaredError()]'_** pada model.compile dengan cara kerja  RMSE dihitung dengan mengkuadratkan error (prediksi – observasi) dibagi dengan jumlah data (= rata-rata), lalu diakarkan.

   hasil dari model evaluasi visualisasi matriks adalah sebagai berikut :  
   ![result_Evaluation](img\result_Evaluation.png)

   dari visualisasi proses training model di atas cukup baik dan model konvergen pada epochs sekitar 100. Dari proses tersebut, memperoleh nilai error akhir sebesar sekitar 0.19 dan error pada data validasi sebesar 0.20.  


## References
Referensi dari proyek yang dikerjakan adalah sebagai berikut :
[Sistem Rekomendasi Film Menggunakan Content Based Filtering](https://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/9163)