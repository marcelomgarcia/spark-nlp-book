from lib2to3.pgen2.token import SLASH
import logging as LG
import os

import pyspark as PK
from pyspark.sql import SparkSession as SS
from pyspark.sql import functions as FF
from pyspark.sql import types as TY

import sparknlp.pretrained as PRETRAIN

def main(spark):

    project_dir = os.getcwd()
    mini_newsgroup_path = os.path.join(project_dir, 'data', 'mini_newsgroups', '*')
    print(mini_newsgroup_path)

    texts = spark.sparkContext.wholeTextFiles(mini_newsgroup_path)

    schema = TY.StructType([
        TY.StructField('filename', TY.StringType()),
        TY.StructField('text', TY.StringType()),
    ])

    LG.info('Creating the dataframe')
    texts_df = spark.createDataFrame(texts, schema)

    LG.info('Printing the first 5 rows of the dataframe')
    texts_df.show(n=5, truncate=100, vertical=True)

    # Hello world from Spark NLP
    LG.info('Starting the NLP part')

    texts_df = texts_df.withColumn(
        'newsgroup', 
        FF.split('filename', '/').getItem(7)
    )

    texts_df.show(5)

    pipeline = PRETRAIN.PretrainedPipeline('explain_document_ml')

    print(pipeline.annotate('hello worldu'))

if __name__ == "__main__":
    spark = SS.builder\
        .appName('Staring to NLP')\
        .master('local[*]')\
        .getOrCreate()

    spark.sparkContext.setLogLevel('WARN')
    main(spark)
    spark.stop()