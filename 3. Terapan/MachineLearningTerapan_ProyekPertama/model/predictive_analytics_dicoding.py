# -*- coding: utf-8 -*-
"""Predictive Analytics_Dicoding.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1f2eDSU9-2bWSahb9aD180YzvSGVsJwPq
"""

from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

drive.mount('/content/drive/')

"""# DataFrame

https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory?select=coin_Bitcoin.csv
"""

df = pd.read_csv('/content/drive/MyDrive/colab_data/coin_Bitcoin.csv')
df.head(10)

"""## Exploratory Data

### Variabel pada dastaset :

1. Name : Nama mata uang kripto
2. Symbol : Simbol mata uang
3. Date : Tanggal pencatatan data
4. High : Harga tertinggi pada hari tertentu
5. Low : Harga terendah pada hari tertentu
6. Open : Harga pembukaan pada hari tertentu
7. Close : Harga penutupan pada hari tertentu
8. Volume : Volume transaksi pada hari tertentu
9. Mastercap : Kapitalisasi pasar dalam USD

## Informasi Data
"""

df.info()

df.describe()

"""## Informasi Statistik

Data di atas memiliki beberapa informasi statistik pada masing-masing kolom, antara lain:

1. count adalah jumlah sampel pada data.
2. mean adalah nilai rata-rata.
3. std adalah standar deviasi.
4. min yaitu nilai minimum setiap kolom.
5. 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama.
6. 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
7. 75% adalah kuartil ketiga.
8. Max adalah nilai maksimum
"""

numeric_features = df.select_dtypes(include=np.number).columns.tolist()
plt.figure(figsize=(15, 8))

for i, col in enumerate(numeric_features):
  plt.subplot(3,3,i+1)
  df.boxplot(column=col)

"""Menerapkan teknik IQR Method yaitu dengan menghapus data yang berada diluar interquartile range. Interquartile merupakan range diantara kuartil pertama(25%) dan kuartil ketiga(75%)."""

Q1 = df.quantile(.25)
Q3 = df.quantile(.75)

IQR = Q3 - Q1

bot_treshold = Q1 - 1.5 * IQR
top_treshold = Q3 + 1.5 * IQR

df = df[~((df < bot_treshold) | (df > top_treshold)).any(axis=1)]
df.shape

"""## Univariate Analysis

Karena target prediksi dari dataset ini ada pada fitur Close_Price yang merupakan harga crypto coin Ethereum, jadi hanya fokus menganalisis korelasi data pada feature tersebut. Dari hasil visualisasi data dibawah dapat disimpulkan bahwa peningkatan harga crypto coin ethereum sebanding dengan penurunan jumlah sampel data.
"""

df.hist(bins=50, figsize=(15, 10))
plt.show()

"""## Multivariate Analysis

Fitur Close pada sumbu y memiliki korelasi dengan data pada fitur High, Low, Open, dan Marketcap. Korelasi yang terdapat pada data-data tersebut merupakan korelasi yang tinggi, sedangkan untuk fitur Volume terlihat memiliki korelasi yang cukup lemah karena sebaran datanya tidak membentuk pola.
"""

sns.pairplot(df, diag_kind = 'kde')
plt.show()

"""Untuk lebih jelasnya dapat dilihat melalui visualisasi dibawah yang menunjukkan skor korelasi di tiap fitur dengan fitur Close. Pada fitur High, Low, Open dan Marketcap memiliki skor korelasi yang terbilang tinggi yaitu 1.00. Sedangkan pada fitur Volume memiliki skor korelasi yang cukup rendah yaitu 0.78. Sehingga fitur Volume ini dapat didrop dari dataset."""

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr().round(2)
 
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

df.drop(['Volume'], axis=1, inplace=True)
df

"""# Data Preparation

## Menghapus data dan merubah nama kolom
Data yang digunakan hanya kolom High, Low, Open, dan Close. Kemudian mengubah nama kolom tersebut seperti yang ada di bawah.
"""

unused_columns = ['SNo', 'Name', 'Symbol', 'Date', 'Marketcap']
renamed_columns = {'High': 'High_Price', 'Low': 'Low_Price', 
                   'Open': 'Open_Price', 'Close': 'Close_Price'}

df.drop(unused_columns, axis=1, inplace=True)
df.rename(columns=renamed_columns, inplace=True)
df

"""## Membagi dataset
Data dibagi menjadi training dan testing
"""

x = df.drop(['Close_Price'], axis=1).values
y = df['Close_Price'].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)

print(f'Total # of sample in whole dataset: {len(x)}')
print(f'Total # of sample in train dataset: {len(x_train)}')
print(f'Total # of sample in test dataset: {len(x_test)}')

"""## Normalisasi data

Menggunakan MinMaxScaler untuk mentransformasikan ditur dengan mengskalakan setiap fitur ke rentang tertentu.

"""

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)

models = pd.DataFrame(columns=['train_mse', 'test_mse'],
                      index=['KNN', 'RandomForest', 'SVR'])

"""# Model Development

## Tuning Hyperparameters

Untuk emndapatkan parameter dengan peforma terbaik pada model maka dilakukan Tuning Hyperparameters dengan menggunakan teknik Grid Search. Grid Search memungkinkan untuk menguji beberapa parameter sekaligus dalam sebuah model dan diharapkan dapat melihat peforma terbaik dengan parameter tertentu.
"""

svr = SVR()
parameters = {
    'kernel': ['rbf'],
    'C':     [1000, 10000, 100000],
    'gamma': [0.3, 0.03, 0.003]
}

svr_search = GridSearchCV(
    svr, 
    parameters,
    cv=5, 
    verbose=1,
    n_jobs=6,
)

svr_search.fit(x_train, y_train)
svr_best_params = svr_search.best_params_

knn = KNeighborsRegressor()
parameters =  {
    'n_neighbors': range(1, 25),
}

knn_search = GridSearchCV(
  knn, 
  parameters, 
  cv=5,
  verbose=1, 
  n_jobs=6,
)

knn_search.fit(x_train, y_train)
knn_best_params = knn_search.best_params_

rf = RandomForestRegressor()
parameters =  {
    'n_estimators': range(1, 10),
    'max_depth': [16, 32, 64],
}

rf_search = GridSearchCV(
  rf, 
  parameters, 
  cv=5,
  verbose=1,
  n_jobs=6,
)
rf_search.fit(x_train, y_train)
rf_best_params = rf_search.best_params_

"""## Training Model

### Support Vector Machine
"""

svr = SVR(
  C=svr_best_params["C"], 
  gamma=svr_best_params["gamma"], 
  kernel=svr_best_params['kernel']
)                          
svr.fit(x_train, y_train)

"""### K-Nearest Neighbours"""

knn = KNeighborsRegressor(n_neighbors=knn_best_params["n_neighbors"])
knn.fit(x_train, y_train)

"""### Random Forest"""

rf = RandomForestRegressor(
  n_estimators=rf_best_params["n_estimators"], 
  max_depth=rf_best_params["max_depth"]
)
rf.fit(x_train, y_train)

"""## Model Evaluation"""

x_test = scaler.transform(x_test)

model_dict = {'KNN': knn, 'RandomForest': rf, 'SVR': svr}

for name, model in model_dict.items():
  models.loc[name, 'train_mse'] = mean_squared_error(
    y_true=y_train, 
    y_pred=model.predict(x_train)
  )
  models.loc[name, 'test_mse'] = mean_squared_error(
    y_true=y_test, 
    y_pred=model.predict(x_test)
  ) 

models

fig, ax = plt.subplots()
models.sort_values(by='test_mse', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)

test_data = x_test.copy()
predictions = {'y_true':y_test}
for name, model in model_dict.items():
  predictions['prediction_' + name] = model.predict(test_data)
 
predictions = pd.DataFrame(predictions)
predictions

predictions = predictions.tail(10)
predictions.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()