from lib2to3.pgen2.token import SLASH
import pyspark as PK

from pyspark.sql import SparkSession as SS
from pyspark.sql import functions as FF
from pyspark.sql import types as TY

import os

def main(spark):

    project_dir = os.getcwd()
    mini_newsgroup_path = os.path.join(project_dir, 'data', 'mini_newsgroups', '*')
    print(mini_newsgroup_path)

    texts = spark.sparkContext.wholeTextFiles(mini_newsgroup_path)

    schema = TY.StructType([
        TY.StructField('filename', TY.StringType()),
        TY.StructField('text', TY.StringType()),
    ])

    texts_df = spark.createDataFrame(texts, schema)

    texts_df.show()

if __name__ == "__main__":
    spark = SS.builder.appName('Staring to NLP').master('local[*]').getOrCreate()

    spark.sparkContext.setLogLevel('WARN')
    main(spark)
    spark.stop()