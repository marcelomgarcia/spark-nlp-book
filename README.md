# Natural Language Processing with Spark NLP

This the repo for the book [Natural Language Processing with Spark NLP: Learning to Understand Text at Scale](https://www.amazon.com/Natural-Language-Processing-Spark-NLP/dp/1492047767)

I have two folders for the notebooks

- `colab` is the folder containing the notebooks to be run on google colab
- `jupyter` is the folder to run the notebooks locally

I'm am working on a docker deployment, and it should be done soon.

There are a couple chapters where you may run into problems

- Chapter 9: I've had some problems running the Core NLP server
- Chapter 13: Uploading the data to Neo4j takes a very long time
- Chapter 14: I've had some problems with spark-elasticsearch interface, also I have not been able to run elasticsearch on colab


## Data

Download the `mini_newsgroups.tar.gz` [data set](https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/)

```
mgarcia@jammy:~/Work/spark-nlp-book$ wget https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/mini_newsgroups.tar.gz
```

and extract it in the project folder

```
mgarcia@jammy:~/Work/spark-nlp-book$ tar zxvf mini_newsgroups.tar.gz -C data/

```

## Spark

Using the Docker image for [pyspark](https://hub.docker.com/r/apache/spark-py). To run the container mounting the data directory as `/mydata`

```
docker run -it --mount type=bind,source="$(pwd)/data",target=/mydata apache/spark-py /opt/spark/bin/pyspark
```

