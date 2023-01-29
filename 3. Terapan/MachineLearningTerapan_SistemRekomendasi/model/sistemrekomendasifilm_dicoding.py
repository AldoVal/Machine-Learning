# -*- coding: utf-8 -*-
"""SistemRekomendasiFilm_Dicoding

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tArxvOwzO7dA9nf95kUR2ornRqfbL_ty

# Sistem Rekomendasi Film
### Aldo Valentino | M180X0298

## Data Understanding

Dataset yang saya gunakan adalah [Dataset Sistem Rekomendasi Movie](https://www.kaggle.com/code/darpan25bajaj/movie-recommendation-system/data)
"""

from google.colab import drive
import pandas as pd
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt

drive.mount('/content/drive/')

import pandas as pd
 
links = pd.read_csv('/content/drive/MyDrive/colab_data/sistem_rekomendasi/ml-latest-small/links.csv')
movies = pd.read_csv('/content/drive/MyDrive/colab_data/sistem_rekomendasi/ml-latest-small/movies.csv')
ratings = pd.read_csv('/content/drive/MyDrive/colab_data/sistem_rekomendasi/ml-latest-small/ratings.csv')
tags = pd.read_csv('/content/drive/MyDrive/colab_data/sistem_rekomendasi/ml-latest-small/tags.csv')

print('Jumlah data link movie : ', len(links.movieId.unique()))
print('Jumlah data movie : ', len(movies.movieId.unique()))
print('Jumlah data ratings dari user : ', len(ratings.userId.unique()))
print('Jumlah data ratings dari user : ', len(ratings.movieId.unique()))
print('Jumlah data : ', len(tags.movieId.unique()))

"""### Univariate Exploratory Data Analysis
Variabel-variabel pada movie-recommendation-data adalah sebagai berikut :

- links : merupakan daftar tautan dari film atau tayangan tersebut.
- movies : merupakan daftar film atau tayangan yang tersedia.
- ratings : merupakan daftar penilaian yang diberikan pengguna terhadap tayangan.
- tags : merupakan daftar kata kunci dari movie tersebut.

####  Links
"""

links.info()

"""####  Movies

"""

movies.info()

"""#### Ratings

"""

ratings.head()

"""cek nilai data dari data ratings"""

ratings.describe()

"""Dari output di atas, diketahui bahwa nilai maksimum ratings adalah 5 dan nilai minimumnya adalah 0.5. Artinya, skala rating berkisar antara 0.5 hingga 5.

## Data Preprocessing


### Menggabungkan Movie
"""

import numpy as np
 
movie_all = np.concatenate((
    links.movieId.unique(),
    movies.movieId.unique(),
    ratings.movieId.unique(),
    tags.movieId.unique(),
))
 
movie_all = np.sort(np.unique(movie_all))
 
print('Jumlah seluruh data movie berdasarkan movieID: ', len(movie_all))

"""### Menggabungkan Seluruh User"""

user_all = np.concatenate((
    ratings.userId.unique(),
    tags.userId.unique(),
   
))
 
user_all = np.sort(np.unique(user_all)) 
 
print('Jumlah seluruh user: ', len(user_all))

"""Menggabungkan file links, movies, ratingsm tags ke dalam dataframe movie_info. Serta menggabungkan dataframe ratings dengan movie_info berdasarkan nilai movieId"""

movie_info = pd.concat([links, movies, ratings, tags])
movie = pd.merge(ratings, movie_info , on='movieId', how='left')
movie

"""Dari hasil diatas terdapat banyak sekali missing value maka lakukan cek missing value

"""

movie.isnull().sum()

"""menggabungkan rating berdasarkan movieId"""

movie.groupby('movieId').sum()

"""### Menggabungkan Data dengan Fitur Nama Movie

mendefinisikan variabel all_movie_rate dengan variabel ratings 
"""

all_movie_rate = ratings
all_movie_rate

"""Menggabungkan all movie_rate dengan dataframe movies berdasarkan movieId """

all_movie_name = pd.merge(all_movie_rate, movies[['movieId','title','genres']], on='movieId', how='left')
all_movie_name

"""Menggabungkan dataframe tags dengan all_movie_name berdasarkan movieId dan memasukkannya ke dalam variabel all_movie"""

all_movie = pd.merge(all_movie_name, tags[['movieId','tag']], on='movieId', how='left')
all_movie

"""## Data Preparation


### Mengatasi Missing Value
"""

all_movie.isnull().sum()

"""Dari data diatas terdapat data kosong pada kolom tag yaitu 52549, maka dilakukanlah pembersihan missing value dengan fungsi dropna()"""

all_movie_clean = all_movie.dropna()
all_movie_clean

"""data di atas berubah menjadi 233213 baris yang awalnya 285762 baris. 
Kemudian periksa kembali missing value pada variabel all_movie_clean

"""

all_movie_clean.isnull().sum()

"""Mengurutkan movie berdasarkan movieId kemudian memasukkannya ke dalam variabel fix_movie"""

fix_movie = all_movie_clean.sort_values('movieId', ascending=True)
fix_movie

"""Mengecek berapa jumlah fix_movie"""

len(fix_movie.movieId.unique())

"""Membuat variabel preparation yang berisi dataframe fix_movie kemudian mengurutkan berdasarkan movieId"""

preparation = fix_movie
preparation.sort_values('movieId')

"""Selanjutnya, gunakan data unik untuk dimasukkan ke dalam proses pemodelan. 
serta hapus data duplicate dengan fungsi drop_duplicates() berdasarkan movieId
"""

preparation = preparation.drop_duplicates('movieId')
preparation

"""Selanjutnya,  melakukan konversi data series menjadi list. Dalam hal ini, menggunakan fungsi tolist() dari library numpy. Implementasikan """

movie_id = preparation['movieId'].tolist()

movie_name = preparation['title'].tolist()
 
movie_genre = preparation['genres'].tolist()
 
print(len(movie_id))
print(len(movie_name))
print(len(movie_genre))

"""membuat dictionary untuk menentukan pasangan key-value pada data movie_id, movie_name, dan movie_genre yang telah disiapkan sebelumnya."""

movie_new = pd.DataFrame({
    'id': movie_id,
    'movie_name': movie_name,
    'genre': movie_genre
})
movie_new

"""## Modeling and Result

1. Model Development dengan Content Based Filtering
<br>
menggukan fungsi TFIDFVectorizer()

"""

from sklearn.feature_extraction.text import TfidfVectorizer
 
# Inisialisasi TfidfVectorizer
tf = TfidfVectorizer()
 
# Melakukan perhitungan idf pada data genre
tf.fit(movie_new['genre']) 
 
# Mapping array dari fitur index integer ke fitur nama
tf.get_feature_names()

"""Selanjutnya, lakukan fit dan transformasi ke dalam bentuk matriks. """

tfidf_matrix = tf.fit_transform(movie_new['genre']) 
tfidf_matrix.shape

""" 
 menghasilkan vektor tf-idf dalam bentuk matriks, menggunakan fungsi todense(). 
"""

tfidf_matrix.todense()

"""lihat matriks tf-idf untuk beberapa movie (movie_name) dan genre"""

pd.DataFrame(
    tfidf_matrix.todense(), 
    columns=tf.get_feature_names(),
    index=movie_new.movie_name
).sample(22, axis=1).sample(10, axis=0)

""" **Cosine Similarity**
 menghitung derajat kesamaan (similarity degree) antar movie dengan teknik cosine similarity.
"""

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix) 
cosine_sim

"""Membuat dataframe dari variabel cosine_sim_df dengan baris dan kolom berupa nama movie, serta melihat kesamaan matrix dari setiap movie"""

cosine_sim_df = pd.DataFrame(cosine_sim, index=movie_new['movie_name'], columns=movie_new['movie_name'])
print('Shape:', cosine_sim_df.shape)
 
cosine_sim_df.sample(5, axis=1).sample(10, axis=0)

"""## Mendapatkan Rekomendasi  



"""

def movie_recommendations(nama_movie, similarity_data=cosine_sim_df, items=movie_new[['movie_name', 'genre']], k=5):
   
 
    # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan    
    # Dataframe diubah menjadi numpy
    # Range(start, stop, step)
    index = similarity_data.loc[:,nama_movie].to_numpy().argpartition(
        range(-1, -k, -1))
    
    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    
    # Drop nama_movie agar nama movie yang dicari tidak muncul dalam daftar rekomendasi
    closest = closest.drop(nama_movie, errors='ignore')
 
    return pd.DataFrame(closest).merge(items).head(k)

"""
 terapkan kode di atas untuk menemukan rekomendasi movie yang mirip dengan Toy Story (1995)."""

movie_new[movie_new.movie_name.eq('Toy Story (1995)')]

"""dari hasil di atas dapat dilihat bahwa pengguna menyukai movie yang berjudul Toy Story (1995)	 yang bergenre Adventure|Animation|Children|Comedy|Fantasy.
Mendapatkan rekomendasi movie yang mirip dengan Toy Story (1995).


"""

movie_recommendations('Toy Story (1995)')

"""Dari hasil rekomendasi di atas, diketahui bahwa Toy Story (1995) termasuk ke dalam genre Adventure|Animation|Children|Comedy|Fantasy. Dari 5 item yang direkomendasikan, 2 item memiliki genre Adventure|Animation|Children|Comedy|Fantasy (similar). Artinya, precision sistem tersebut sebesar 2/5 atau 40%. Dan 3/5 daintaranya memiliki 4 genre yang sama yakni Adventure|Animation|Children|Fantasy.

2. Model Development dengan Collaborative Filtering
"""

import pandas as pd
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt

"""ubah nama variabel ratings yang telah dibuat sebelumnya menjadi df."""

df = ratings
df

"""## Data Preparation
melakukan tahapan prepocessing
"""

# Mengubah userID menjadi list tanpa nilai yang sama
user_ids = df['userId'].unique().tolist()
print('list userID: ', user_ids)
 
# Melakukan encoding userID
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
print('encoded userID : ', user_to_user_encoded)
 
# Melakukan proses encoding angka ke ke userID
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
print('encoded angka ke userID: ', user_encoded_to_user)

"""Selanjutnya, lakukan hal yang sama pada fitur ‘movieId’."""

# Mengubah movieId menjadi list tanpa nilai yang sama
movie_ids = df['movieId'].unique().tolist()
 
# Melakukan proses encoding movieId
movie_to_movie_encoded = {x: i for i, x in enumerate(movie_ids)}
 
# Melakukan proses encoding angka ke movieId
movie_encoded_to_movie = {i: x for i, x in enumerate(movie_ids)}
 
# Selanjutnya, petakan userId dan movieId ke dataframe yang berkaitan.
 
# Mapping userId ke dataframe genres
df['genres'] = df['userId'].map(user_to_user_encoded)
 
# Mapping movieD ke dataframe movies
df['movies'] = df['movieId'].map(movie_to_movie_encoded)

"""Terakhir, cek beberapa hal dalam data seperti jumlah user, jumlah movie, dan mengubah nilai rating menjadi float, cek nilai minimum dan maximum"""

num_users = len(user_to_user_encoded)
print(num_users)
 
num_movie = len(movie_encoded_to_movie)
print(num_movie)
 
df['ratings'] = df['rating'].values.astype(np.float32)
 
min_rating = min(df['rating'])
 
max_rating = max(df['rating'])
 
print('Number of User: {}, Number of movie: {}, Min Rating: {}, Max Rating: {}'.format(
    num_users, num_movie, min_rating, max_rating
))

"""**Membagi Data untuk Training dan Validasi**

"""

df = df.sample(frac=1, random_state=42)
df

"""membagi data train dan validasi dengan komposisi 80:20. """

x = df[['genres', 'movies']].values
 
y = df['ratings'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)
 
print(x, y)

"""lakukan proses training

"""

class RecommenderNet(tf.keras.Model):
 
  # Insialisasi fungsi
  def __init__(self, num_users, num_movie, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_movie = num_movie
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding( # layer embedding user
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1) # layer embedding user bias
    self.movie_embedding = layers.Embedding( # layer embeddings movies
        num_movie,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.movie_bias = layers.Embedding(num_movie, 1) # layer embedding movies bias
 
  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
    user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
    movie_vector = self.movie_embedding(inputs[:, 1]) # memanggil layer embedding 3
    movie_bias = self.movie_bias(inputs[:, 1]) # memanggil layer embedding 4
 
    dot_user_movie = tf.tensordot(user_vector, movie_vector, 2) 
 
    x = dot_user_movie + user_bias + movie_bias
    
    return tf.nn.sigmoid(x) # activation sigmoid

"""## Evaluation 
Selanjutnya, lakukan proses compile terhadap model. serta menggunakan matrix evaluasi RMSE


"""

model = RecommenderNet(num_users, num_movie, 50)
 
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

"""Memulai proses training dengan batch size sebesar 64 serta epoch 100 kali"""

history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 64,
    epochs = 100,
    validation_data = (x_val, y_val)
)

"""**Visualisasi Metrik**  
Untuk melihat visualisasi proses training
"""

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

"""Dari visualisasi proses training model di atas cukup baik dan model konvergen pada epochs sekitar 100. Dari proses tersebut, memperoleh nilai error akhir sebesar sekitar 0.19 dan error pada data validasi sebesar 0.20.

***Mendapatkan Rekomendasi movie***
"""

movie_df = movie_new
df = pd.read_csv('/content/drive/MyDrive/colab_data/sistem_rekomendasi/ml-latest-small/ratings.csv')
 

user_id = df.userId.sample(1).iloc[0]
movie_watched_by_user = df[df.userId == user_id]
 

movie_not_watched = movie_df[~movie_df['id'].isin(movie_watched_by_user.movieId.values)]['id'] 
movie_not_watched = list(
    set(movie_not_watched)
    .intersection(set(movie_to_movie_encoded.keys()))
)
 
movie_not_watched = [[movie_to_movie_encoded.get(x)] for x in movie_not_watched]
user_encoder = user_to_user_encoded.get(user_id)
user_movie_array = np.hstack(
    ([[user_encoder]] * len(movie_not_watched), movie_not_watched)
)

"""Agar memperoleh rekomendasi movies, gunakan fungsi model.predict() dari library Keras dengan menerapkan kode berikut."""

ratings = model.predict(user_movie_array).flatten()
 
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_movie_ids = [
    movie_encoded_to_movie.get(movie_not_watched[x][0]) for x in top_ratings_indices
]
 
print('Showing recommendations for users: {}'.format(user_id))
print('===' * 9)
print('movie with high ratings from user')
print('----' * 8)
 
top_movie_user = (
    movie_watched_by_user.sort_values(
        by = 'rating',
        ascending=False
    )
    .head(5)
    .movieId.values
)
 
movie_df_rows = movie_df[movie_df['id'].isin(top_movie_user)]
for row in movie_df_rows.itertuples():
    print(row.movie_name, ':', row.genre)
 
print('----' * 8)
print('Top 10 movie recommendation')
print('----' * 8)
 
recommended_movie = movie_df[movie_df['id'].isin(recommended_movie_ids)]
for row in recommended_movie.itertuples():
    print(row.movie_name, ':', row.genre)