# Invoice: Fatura numarası
# StockCode: Ürün kodu
# Description: Ürün açıklaması
# Quantity: Satılan miktar
# InvoiceDate: Fatura tarihi
# Price: Ürün fiyatı
# Customer ID: Müşteri numarası
# Country: Müşteri ülkesi
import pandas as pd

#veri setinde 2009-2011 arası işlemler bulunmakta biz 2010-2011 bakacağız ve veri seti çok büyük olduğu için genel 50000 satır ile ilgileneceğim


import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from numpy.lib.function_base import average

##########################################
##########################################
#1. RFM Analizi (Recency, Frequency, Monetary)
##########################################
##########################################


##########################################
#1-Veri Hazırlama
##########################################

# Recency (Güncellik): Son alışverişinden kaç gün geçti?
# Frequency (Sıklık): Kaç farklı alışveriş yaptı?
# Monetary (Parasal Değer): Toplam harcama miktarı nedir?

# Dosya okunuyor
df = pd.read_excel(r"C:\Users\PC\OneDrive\Masaüstü\veribilimi\verisetleri\online+retail+ii\online_retail_II.xlsx",
                   sheet_name="Year 2010-2011",
                   nrows=50000)

# Customer ID'si boş olmayanları filtrele
df = df[df["Customer ID"].notnull()]

# İlk 5 satırı göster
print(df.head())

##########################################
#2-RFM Değerlerini Hesaplayalım
##########################################

# Veri setindeki en son tarihi bulalım
max_date = df["InvoiceDate"].max()
print(max_date)

#RFM hesaplamaları
rfm_df = df.groupby("Customer ID").agg({
    "InvoiceDate": lambda x: (max_date - x.max()).days, # Recency (Son alışverişin üzerinden geçen gün)
    "Invoice": "nunique", # Frequency (Kaç farklı alışveriş yapmış)
    "Price": lambda x: (x * df.loc[x.index, "Quantity"]).sum() # Monetary (Toplam harcama)
})

# Sütun adlarını güncelle
rfm_df.columns = ["Recency", "Frequency", "Monetary"]

# Negatif veya sıfır monetary değerleri çıkar
rfm_df = rfm_df[rfm_df["Monetary"] > 0]

# İlk 5 satırı göster
print(rfm_df.head())



##########################################
#3-RFM Skorlarını Oluşturma
##########################################

# Recency, Frequency ve Monetary skorlarını hesapla
rfm_df["R_Score"] = pd.qcut(rfm_df["Recency"], 4, labels=[4,3,2,1])
rfm_df["F_Score"] = pd.qcut(rfm_df["Frequency"].rank(method="first"), 4, labels=[1,2,3,4])
rfm_df["M_Score"] = pd.qcut(rfm_df["Monetary"].rank(method="first"), 4, labels=[1,2,3,4])

# RFM Skorlarını birleştir
rfm_df["RFM_Score"] = rfm_df["R_Score"].astype(str) + rfm_df["F_Score"].astype(str) + rfm_df["M_Score"].astype(str)

# İlk 5 satırı göster
print(rfm_df.head())


##########################################
#3-Müşteri Segmentasyonu
##########################################

def segment_rfm(score):
    if score in ["444", "434", "344"]:
        return "VIP Müşteriler"
    elif score in ["411", "311", "211"]:
        return "Yeni Müşteriler"
    elif score in ["144", "134", "124"]:
        return "Sadık Müşteriler"
    elif score in ["244", "233", "222"]:
        return "Potansiyel Sadık"
    elif score in ["111", "112", "121"]:
        return "Kaybedilen Müşteriler"
    else:
        return "Orta Seviye Müşteri"

# Segment sütunu ekle
rfm_df["Segment"] = rfm_df["RFM_Score"].apply(segment_rfm)

# Segmentlerin dağılımını göster
print(rfm_df["Segment"].value_counts())



##########################################
##########################################
#2. Müşteri Yaşam Boyu Değeri (CLV) Hesaplama
##########################################
##########################################

#Şimdi her müşterinin yaşam boyu değerini (CLV) hesaplayacağız.

#Ortalama Sipariş Değeri (AOV)
total_revenue = df["Quantity"] * df["Price"]
total_orders = df["Invoice"].nunique()
aov = total_revenue.sum() / total_orders

#Satın Alma Sıklığı (Purchase Frequency)
total_customers = df["Customer ID"].nunique()
purchase_frequency = total_orders / total_customers

#Müşteri Değeri (Customer Value)
customer_value = aov * purchase_frequency

#Müşteri Ortalama Yaşam Süresi (Customer Lifespan)
customer_recency = df.groupby("Customer ID")["InvoiceDate"].max().apply(lambda x: (max_date - x).days)
average_lifespan = customer_recency.mean() / 365

#Müşteri Yaşam Boyu Değeri (CLV)
clv = customer_value * average_lifespan

print(f"AOV: {aov}, Purchase Frequency: {purchase_frequency}, CLV: {clv}")


##########################################
##########################################
#3. CLV Tahmini (Regresyon Modeli)
##########################################
##########################################

#Müşterinin gelecekteki harcama tahminini yapmak için Makine Öğrenmesi modeli kuracağız.

# Bağımsız değişkenler (x1) ve bağımlı değişken (y1)
x1 = rfm_df[["Recency", "Frequency", "Monetary"]]
y1 = rfm_df["Monetary"]

# Eğitim ve test verisini ayır
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
model = LinearRegression()
model.fit(x1_train, y1_train)

# Model doğruluk skoru
print(f"Model Doğruluk Skoru: {model.score(x1_test, y1_test)}")

# Örnek bir müşterinin gelecekteki harcamasını tahmin et
sample_customer = [[30, 2, 500]] # Recency: 30 gün önce, Frequency: 2 alışveriş, Monetary: 500$
predicted_value = model.predict(sample_customer)
print(f"Tahmini CLV: {predicted_value[0]}")












