#!/usr/bin/env python
# coding: utf-8

# # TOPIC MODELLING LSA

# # Tahap Crawling Data

# Crawling adalah teknik pengumpulan data yang digunakan untuk mengindeks informasi pada halaman menggunakan URL (Uniform Resource Locator) dengan menyertakan API (Application Programming Interface) untuk melakukan penambangan dataset yang lebih besar.

# Melakukan Scrapy
# 
# Scrapy ini adalah library python yang digunakan untuk melakukan scraping/ crawling data

# In[1]:


get_ipython().system('pip install scrapy')


# In[2]:


import scrapy
from scrapy.crawler import CrawlerProcess


# Pada bagian ini dilakukan proses crawling pada data web berita. Lalu memasukkan url yang dituju. 
# 
# Disini dilakukan crawling pada web sindonews dengan mengambil data berupa judul, waktu, kategori, dan deskripsi.

# In[3]:


class ScrapingWeb(scrapy.Spider):    
    name = "sindonews"
    keyword = 'edukasi'
    start_urls = [
        'https://edukasi.sindonews.com/'+keyword
        ]
    custom_settings = {
        'FEED_FORMAT': 'csv',
        'FEED_URI': 'crawlingweb.csv'
        }
    
    def parse(self, response):
        for data in response.css('div.sinfix'):
            yield {
                'judul': data.css('div.title a::text').get(),
                'waktu': data.css('time::text').get(),
                'kategori':data.css('span::text').get(),
                'deskripsi': data.css('div.subcaption::text').get()
                }
proses = CrawlerProcess()
proses.crawl(ScrapingWeb)
proses.start()


# # Melakukan Import Module

# Terdapat beberapa library yang harus diimport terlebih dahulu.
# - Numpy: NumPy (Numerical Python) adalah library Python yang fokus pada scientific computing dan memiliki kemampuan untuk membentuk objek N-dimensional array, yang mirip dengan list pada Python. Keunggulan NumPy array dibandingkan dengan list pada Python adalah konsumsi memory yang lebih kecil serta runtime yang lebih cepat.
# 
# - Pandas (Python for Data Analysis) adalah library Python yang fokus untuk proses analisis data seperti manipulasi data, persiapan data, dan pembersihan data. Pandas menyediakan struktur data dan fungsi high-level untuk membuat data lebih terstruktur, lebih cepat, mudah. Dalam pandas terdapat dua objek yang sering dibahas, yaitu DataFrame dan Series.
# 
# - matplotlib.pyplot adalah kumpulan fungsi yang membuat beberapa perubahan pada gambar. misalnya membuat gambar, membuat area plot dalam gambar, menambah label di plot dan lainnya. Biasanya untuk mempermudah secara umum matplotlib.pyplot disingkat menjadi plt import matplotlib.pyplot as plt
# 
# - Matplotlib adalah library Python yang fokus pada visualisasi data yang biasa difungsikan membuat plot grafik. matplotlib tersebut dipanggil dan dilakukan import style.
# 
# - Seaborn memiliki banyak fungsi untuk visualisasi data dan lebih mudah digunakan untuk menggunakan library seaborn kita harus install library ini terlebih dahulu dengan menggunakan PIP pip install seaborn Untuk menggunakannya import terlebih dahulu dengan perintah seperti ini import seaborn as sns.

# In[4]:


# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns


# Melakukan Konfigurasi
# 
# set matplotlib ke inline dan menampilkan grafik di bawah sel yang sesuai.

# In[5]:


#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)


# # Persiapan Preprocessing

# Preprocessing Data merupakan tahapan dalam melakukan mining data. Data Preprocessing atau praproses data biasanya dilakukan melalui cara eliminasi data yang tidak sesuai.
# 
# - Stopwords di nltk adalah kata yang paling umum dalam data. Itu adalah kata-kata yang tidak ingin digunakan untuk menggambarkan topik data dan telah ditentukan sebelumnya juga tidak dapat dihapus.
# 
# - Tokenization pada dasarnya mengacu pada pemisahan teks yang lebih besar menjadi baris yang lebih kecil, kata-kata atau bahkan membuat kata-kata untuk bahasa non-Inggris. 

# In[6]:


#preprocessing
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize,sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet


# In[7]:


# for named entity recognition (NER)
from nltk import ne_chunk


# In[8]:


# vectorizers for creating the document-term-matrix (DTM)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer


# Melakukan instalasi nltk. Nltk (toolkit bahasa alami) adalah salah satu library python suite yang berisi program yang digunakan untuk melakukan pemrosesan bahasa statistik. 

# In[9]:


get_ipython().system('pip install nltk')


# In[10]:


import nltk
nltk.download('stopwords')


# In[11]:


#stop-words
stop_words=set(nltk.corpus.stopwords.words('english'))


# # Menampilkan Dataset

# Disini dilakukan proses untuk menampilkan data yang bernama crawlingweb.csv

# In[12]:


df=pd.read_csv(r'crawlingweb.csv')


# In[13]:


df.head()


# Lakukan drop kolom pada judul, waktu dan kategori, karena yang akan diproses datanya disini adalah deskripsi.

# In[14]:


# drop judul, waktu, kategori
df.drop(['judul','waktu','kategori'],axis=1,inplace=True)


# Menampilkan data berupa 'deskripsi' dengan 26 record data

# In[15]:


df.head(26)


# # Cleaning Data

# Salah satu preprocessing adalah melakukan cleaning data untuk melakukan proses pembersihan data yang tidak diperlukan. Dalam tahap ini, data dibersihkan melalui beberapa proses seperti mengisi nilai yang hilang, menghaluskan noisy data, dan menyelesaikan inkonsistensi yang ditemukan. Data juga bisa dibersihkan dengan dibagi menjadi segmen-segmen yang memiliki ukuran serupa lalu dihaluskan (binning).
# 
# Disini telah menggunakan lemmatizer dan juga dapat menggunakan stemmer. Juga kata-kata berhenti telah digunakan bersama dengan kata-kata yang panjangnya lebih pendek dari 3 karakter untuk mengurangi beberapa kata yang menyimpang. 

# In[16]:


def clean_text(headline):
  le=WordNetLemmatizer()
  word_tokens=word_tokenize(headline)
  tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
  cleaned_text=" ".join(tokens)
  return cleaned_text


# In[17]:


get_ipython().system('pip install nltk')


# In[18]:


import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[19]:


# time taking
df['deskripsi_cleaned_text']=df['deskripsi'].apply(clean_text)


# In[ ]:


df.head(26)


# Drop kolom yang tidak diproses, yang akan didrop/ hapus adalah deskripsi

# In[ ]:


df.drop(['deskripsi'],axis=1,inplace=True)


# # Data Deskripsi_cleaned_text

# Tampilkan data deskripsi_cleaned_text

# In[ ]:


df.head(26)


# In[ ]:


df['deskripsi_cleaned_text'][0]


# MENGEKSTRAKSI FITUR DAN MEMBUAT DOCUMENT-TERM-MATRIX ( DTM )
# Di DTM nilainya adalah nilai TFidf.
# 
# Saya juga telah menentukan beberapa parameter dari vectorizer TF-IDF.
# 
# Berikut poin penting pada LSA
# 
# 1) LSA umumnya diimplementasikan dengan nilai TF-IDF dimana-mana dan bukan dengan Count Vectorizer.
# 
# 2) max_features tergantung pada daya komputasi Anda dan juga pada eval. matrik (skor koherensi adalah matrik untuk model topik). Cobalah nilai yang memberikan eval terbaik. matrik dan tidak membatasi kekuatan pemrosesan.
# 
# 3) Nilai default untuk min_df &max_df bekerja dengan baik.
# 
# 4) Dapat mencoba nilai yang berbeda untuk ngram_range.

# In[ ]:


vect =TfidfVectorizer(stop_words=stop_words,max_features=1000)


# In[ ]:


vect_text=vect.fit_transform(df['deskripsi_cleaned_text'])


# Kita sekarang melihat kata-kata yang paling sering muncul dan langka di berita utama berdasarkan skor idf. Semakin kecil nilainya; lebih umum adalah kata dalam berita utama.

# In[ ]:


print(vect_text.shape)
print(vect_text)


# In[ ]:


idf=vect.idf_


# In[ ]:


dd=dict(zip(vect.get_feature_names(), idf))
l=sorted(dd, key=(dd).get)
# print(l)
print(l[0],l[-1])
print(dd['anak'])
print(dd['aliyah'])


# Oleh karena itu, dapat dilihat bahwa berdasarkan nilai idf, 'anak' adalah kata yang paling sering terklihat. Sedangkan 'aliyah' paling jarang terlihat di antara berita lainnya.

# # Tahap Topic Modelling LSA

# Latent Semantic Analysis (LSA)
# 
# Latent Semantic Analysis (LSA), adalah salah satu teknik dasar dalam pemodelan topik. Ide intinya adalah mengambil matriks dari apa yang kita miliki - dokumen dan istilah - dan menguraikannya menjadi matriks topik dokumen dan matriks istilah topik yang terpisah.
# 
# Pendekatan pertama yang digunakan adalah LSA. LSA pada dasarnya adalah dekomposisi nilai tunggal.
# 
# SVD menguraikan DTM asli menjadi tiga matriks S=U.(sigma).(V.T). Di sini matriks U menunjukkan matriks dokumen-topik sementara (V) adalah matriks topik-term.
# 
# Setiap baris dari matriks U (matriks istilah dokumen) adalah representasi vektor dari dokumen yang sesuai. Panjang vektor ini adalah jumlah topik yang diinginkan. Representasi vektor untuk suku-suku dalam data kami dapat ditemukan dalam matriks V (matriks istilah-topik).
# 
# Jadi, SVD memberi kita vektor untuk setiap dokumen dan istilah dalam data kita. Panjang setiap vektor adalah k. Kami kemudian dapat menggunakan vektor-vektor ini untuk menemukan kata-kata dan dokumen serupa menggunakan metode kesamaan kosinus.
# 
# Kita dapat menggunakan fungsi truncatedSVD untuk mengimplementasikan LSA. Parameter n_components adalah jumlah topik yang ingin kita ekstrak. Model tersebut kemudian di fit dan ditransformasikan pada hasil yang diberikan oleh vectorizer.
# 
# Terakhir perhatikan bahwa LSA dan LSI (I untuk pengindeksan) adalah sama dan yang terakhir kadang-kadang digunakan dalam konteks pencarian informasi.

# In[ ]:


from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[ ]:


print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[ ]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)


# Demikian pula untuk dokumen lain kita bisa melakukan proses tersebut. Namun perhatikan bahwa nilai tidak menambah 1 seperti di LSA itu bukan kemungkinan topik dalam dokumen.

# In[ ]:


print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)


# Pada proses ini mendapatkan daftar kata-kata penting untuk masing-masing dari 10 topik seperti yang ditunjukkan. Untuk proses tahapannya disini ditunjukkan 10 kata untuk setiap topik.

# In[ ]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")

