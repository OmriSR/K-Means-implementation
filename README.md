# K-Means-implementation

Clustering of text, whether it be sentences, paragraphs, or documents, is a common task in Natural Language Processing (NLP). Unlike labeled datasets, unlabeled data is more abundant and easily accessible. Gaining insights about the data's nature without relying on supervision is crucial and highly valuable.

## Part #1: K-Means Clustering Algorithm Implementation with Random Initialization

This project focuses on implementing the K-Means clustering algorithm with random initialization. Several datasets are provided with annotated categories; however, these labels should only be used for evaluation purposes. During clustering, you should not make use of these annotations.

You will find two files included in this repository:

1. `config.json`: This JSON file contains the location of the datafile and the encoding mode. While you can change the datafile location in `config.json`, please note that you should not modify the code parsing this file in the main script.
2. `main.py`: The main Python script where you need to implement the `kmeans_cluster_and_evaluate()` function. The encoding type varies between "TFIDF" and "SBERT" and is specified in the `config.json` file.

Your primary task is to complete the `kmeans_cluster_and_evaluate()` function while considering the specified encoding type. Please ensure that your implementation adheres to the provided guidelines and the runtime requirements mentioned below.

Feel free to explore different approaches and strategies to improve the performance of the K-Means clustering algorithm. You can leverage popular libraries and tools in the NLP ecosystem to assist you in this task.

### Runtime Requirements

Make sure that your code executes within a reasonable timeframe. While there are no strict limits, it is recommended to optimize your implementation to ensure efficient runtime.

Please document your findings, observations, and any modifications made to the provided guidelines. This documentation will be essential for evaluating your project and understanding your approach.

Good luck with your K-Means clustering implementation!
