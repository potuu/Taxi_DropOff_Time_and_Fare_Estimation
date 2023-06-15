# Databricks notebook source
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.functions import isnull, when, count, col

# COMMAND ----------

# MAGIC %md
# MAGIC ## Veri Setinin Yüklenmesi/ Loading the Data Set

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/yellow_tripdata_2023_01.parquet"
file_type = "parquet"

df = spark.read.format(file_type) \
  .load(file_location)

display(df)


# COMMAND ----------

# MAGIC %md
# MAGIC ## **Veri Ön İşleme**//Data Preprocessing

# COMMAND ----------

# MAGIC %md
# MAGIC ### Yolcu Alış ve Bırakış Sürelerinin Tipinin Dönüştürülmesi // Conversion of Passenger Pickup and Drop-off Times

# COMMAND ----------

df = df.withColumn('tpep_pickup_datetime', col('tpep_pickup_datetime').cast('timestamp'))
df = df.withColumn('tpep_dropoff_datetime', col('tpep_dropoff_datetime').cast('timestamp'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Veri Kümesinin Şemasının Yazdırılması // Printing the Diagram of the Dataset

# COMMAND ----------

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Null Değerlerin Tespiti // Detection of Null Values

# COMMAND ----------

for c in df.columns:
  print("`{:s}` satırındaki null değer sayısı = {:d}".format(c, df.where(col(c).isNull()).count()))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Store_and_fwd_flag ve VendorID Sütunlarının Silinmesi // Deleting Store_and_fwd_flag and VendorID Columns
# MAGIC  
# MAGIC Store_and_fwd_flag sütunu verinin araba hafızasında depolanıp daha sonra gönderilip gönderilmediği bilgisini içeriyor. Bu sütun hem tek kategorik sütun olması hem de bizim amacımız için anlamsız olduğundan silinmiştir. 
# MAGIC
# MAGIC VendorID sütunu ise veriyi kaydeden sağlayıcının id'sini gösterdiğinden silinmiştir.

# COMMAND ----------

df = df.drop('store_and_fwd_flag', 'VendorID')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Null Değerlerin Silinmesi // Deleting Null Values
# MAGIC
# MAGIC Bizim veri setimizde 3 milyon civarında satır ve yalnızca 71743 satır (%2,33) null değer içerdiğinden null değer olan satırlar çıkarılmıştır.

# COMMAND ----------

df = df.dropna()

# Null değer olmadığının gösterilmesi
df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verinin İstatistiksel Özelliklerinin Görüntülenmesi // Displaying Statistical Characteristics of Data

# COMMAND ----------

df.describe().toPandas().transpose()  # Pandas DataFrame'e çevrilip transpose ile satır sütünların yeri değiştirilir.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Yolculuk Süresinin Hesaplanması ve Sıfır veya Negatif Olan Yolculuklarının Silinmesi // Calculating Journey Time and Deleting Zero or Negative Journeys

# COMMAND ----------

df = df.withColumn('trip_duration', col('tpep_dropoff_datetime').cast('long') - col('tpep_pickup_datetime').cast('long'))
df = df.filter(col('trip_duration') > 0)
df.filter(col('trip_duration') <= 0).show()  # Sıfır veya negatif bir değer kalmadığının gösterilmesi

# COMMAND ----------

# MAGIC %md
# MAGIC ### Toplam Ücretin Negatif Olduğu Sütunların Görüntülenmesi ve Çıkarılması // Displaying and Removing Columns with Negative Total Fee

# COMMAND ----------

print("Toplam Ücretin Negatif Olduğu Satır Sayısı:", df.filter(df["total_amount"] < 0).count())
print("Taksimetre Ücretinin Negatif Olduğu Satır Sayısı: ", df.filter(df["fare_amount"] < 0).count())
df.filter((df["total_amount"] < 0) | (df["fare_amount"] < 0)).show()

df = df.filter(df.total_amount >= 0)
df = df.filter(df.fare_amount >=0)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Taksinin Hızının Hesaplanması // Calculating the Speed ​​of the Taxi

# COMMAND ----------

df = df.withColumn('speed', round(col('trip_distance') / (col('trip_duration') / 3600), 2))
df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Yapılan Ön İşlemlerden Sonra İstatistiksel Özelliklerinin Yeniden Görüntülenmesi // Re-Displaying Statistical Properties After Pre-Processing

# COMMAND ----------

df.describe().toPandas().transpose() # Pandas DataFrame'e çevrilip transpose ile satır sütünların yeri değiştirilir.

# COMMAND ----------

# Let's define some constants which we will use throughout this notebook
NUMERICAL_FEATURES = ["passenger_count",
                      "trip_distance",
                      "RatecodeID",
                      "PULocationID",
                      "DOLocationID",
                      "payment_type",
                      "fare_amount",
                      "extra",
                      "mta_tax",
                      "tip_amount",
                      "tolls_amount",
                      "improvement_surcharge",
                      "total_amount",
                      "congestion_surcharge",
                      "airport_fee",
                      "trip_duration",
                      "speed"]

CATEGORICAL_FEATURES = []

TARGET_VARIABLE = "total_amount"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Veri Dağılımlarının Analizi: Sayısal Özellikler // Analysis of Data Distributions: Numerical Properties

# COMMAND ----------

# MAGIC %md
# MAGIC Her sütunun değerlerinin dağılımının çizilmesi

# COMMAND ----------

pdf = df.toPandas()

n_rows = 5
n_cols = 17

# COMMAND ----------

fig, axes = plt.subplots(n_rows, n_cols, figsize=(28,20))

# COMMAND ----------

for i,f in enumerate(NUMERICAL_FEATURES):
    _ = sns.distplot(pdf[f],
                    kde_kws={"color": "#ca0020", "lw": 1}, 
                    hist_kws={"histtype": "bar", "edgecolor": "k", "linewidth": 1,"alpha": 0.8, "color": "#92c5de"},
                    ax=axes[i]
                    )
    

# COMMAND ----------

fig.tight_layout(pad=1.5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### İkili regresyon grafikleri // Binary regression charts

# COMMAND ----------

_ = sns.pairplot(data=pdf, 
                 vars=sorted(NUMERICAL_FEATURES), 
                 hue=TARGET_VARIABLE, 
                 kind="reg",
                 diag_kind='hist',
                 diag_kws = {'alpha':0.55, 'bins':20},
                 markers=["s","X","+"]
                )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verilerin Eğitim ve Test Olarak Ayrılması // Separation of Data as Training and Testing

# COMMAND ----------

RANDOM_SEED = 42

train_df, test_df = df.randomSplit([0.8, 0.2], seed=RANDOM_SEED)

print("Training set size: {:d} instances".format(train_df.count()))
print("Test set size: {:d} instances".format(test_df.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sayısal Verilerin Vektöre Çevrilmesi // Converting Digital Data to Vector

# COMMAND ----------

def to_numerical(df, numerical_features, categorical_features, target_variable):
    """
    Args:
        - df: the input dataframe
        - numerical_features: the list of column names in `df` corresponding to numerical features
        - categorical_features: the list of column names in `df` corresponding to categorical features
        - target_variable: the column name in `df` corresponding to the target variable
    Return:
        - transformer: the pipeline of transformation fit to `df` (for future usage)
        - df_transformed: the dataframe transformed according to the pipeline
    """
    
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

    # 1. Create a list of indexers, i.e., one for each categorical feature
    indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c), handleInvalid="keep") for c in categorical_features]

    # 2. Create the one-hot encoder for the list of features just indexed (this encoder will keep any unseen label in the future)
    encoder = OneHotEncoder(inputCols=[indexer.getOutputCol() for indexer in indexers], 
                                    outputCols=["{0}_encoded".format(indexer.getOutputCol()) for indexer in indexers], 
                                    handleInvalid="keep")

    # 3. Indexing the target column (i.e., transform it into 0/1) and rename it as "label"
    # Note that by default StringIndexer will assign the value `0` to the most frequent label, which in the case of `deposit` is `no`
    # As such, this nicely resembles the idea of having `deposit = 0` if no deposit is subscribed, or `deposit = 1` otherwise.
    label_indexer = StringIndexer(inputCol = target_variable, outputCol = "label")
    
    # 4. Assemble all the features (both one-hot-encoded categorical and numerical) into a single vector
    assembler = VectorAssembler(inputCols=encoder.getOutputCols() + numerical_features, outputCol="features")

    # 5. Populate the stages of the pipeline
    stages = indexers + [encoder] + [label_indexer] + [assembler]

    # 6. Setup the pipeline with the stages above
    pipeline = Pipeline(stages=stages)

    # 7. Transform the input dataframe accordingly
    transformer = pipeline.fit(df)
    df_transformed = transformer.transform(df)

    # 8. Eventually, return both the transformed dataframe and the transformer object for future transformations
    return transformer, df_transformed

# COMMAND ----------

oh_transformer, oh_train_df = to_numerical(train_df, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_VARIABLE)

# COMMAND ----------

train = oh_train_df.select([TARGET_VARIABLE, "features"])

# COMMAND ----------

train.show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## **Eğitim** / Training

# COMMAND ----------

# MAGIC %md
# MAGIC #### Lineer Regresyon // Linear Regression

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

log_reg = LinearRegression(featuresCol="features", labelCol=TARGET_VARIABLE, maxIter=100, regParam=0.1, elasticNetParam=1)
log_reg_model = log_reg.fit(train)

# COMMAND ----------

trainingSummary = log_reg_model.summary

rmse = trainingSummary.rootMeanSquaredError
mse = trainingSummary.meanSquaredError
r2 = trainingSummary.r2

print("Ortalama Karesel Hata:" ,'{:.20f}'.format(mse))
print("Kök Ortalama Karesel Hatası:", '{:.20f}'.format(rmse))
print("R2 Skoru:", '{:.20f}'.format(r2))

# COMMAND ----------

oh_test_df = oh_transformer.transform(test_df)

# COMMAND ----------

oh_test_df.show(5)

# COMMAND ----------

test = oh_test_df.select(["features", TARGET_VARIABLE])
test.show(5)

# COMMAND ----------

predictions = log_reg_model.transform(test)

# COMMAND ----------

predictions.select("features", "prediction", TARGET_VARIABLE).show(10)

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(predictionCol="prediction", labelCol=TARGET_VARIABLE, metricName="rmse")
test_rmse = evaluator.evaluate(predictions)

evaluator = RegressionEvaluator(predictionCol="prediction", labelCol=TARGET_VARIABLE, metricName="mse")
test_mse = evaluator.evaluate(predictions)

evaluator = RegressionEvaluator(predictionCol="prediction", labelCol=TARGET_VARIABLE, metricName="r2")
test_r2 = evaluator.evaluate(predictions)

print("Test MSE:", '{:.20f}'.format(test_mse))
print("Test RMSE:", '{:.20f}'.format(test_rmse))
print("Test R2:", '{:.20f}'.format(test_r2))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Random Forest

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor

# Random Forest modelinin Eğitilmesi
rf = RandomForestRegressor(featuresCol="features", labelCol=TARGET_VARIABLE, numTrees=20, maxDepth=5)
rf_model = rf.fit(train)

# COMMAND ----------

# Test verisi üzerinde tahmin yapılması
predictions = rf_model.transform(test)

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(predictionCol="prediction", labelCol=TARGET_VARIABLE, metricName="rmse")
test_rmse = evaluator.evaluate(predictions)

evaluator = RegressionEvaluator(predictionCol="prediction", labelCol=TARGET_VARIABLE, metricName="mse")
test_mse = evaluator.evaluate(predictions)

evaluator = RegressionEvaluator(predictionCol="prediction", labelCol=TARGET_VARIABLE, metricName="r2")
test_r2 = evaluator.evaluate(predictions)

print("Test MSE:", '{:.20f}'.format(test_mse))
print("Test RMSE:", '{:.20f}'.format(test_rmse))
print("Test R2:", '{:.20f}'.format(test_r2))
