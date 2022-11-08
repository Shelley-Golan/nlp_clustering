import os
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_20newsgroups

from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt

import seaborn as sns

np.random.seed(477)

def create_train_test():
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')
    texts_train = newsgroups_train.data # Extract text
    texts_test = newsgroups_test.data
    return texts_train, texts_test

def vectorize(texts):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)


def elbow_method(X):
    Sum_of_squared_distances = []
    K = range(1,20)
    for k in K:
        km = KMeans(init="k-means++", n_clusters=k)
        km = km.fit(X)
        Sum_of_squared_distances.append(km.inertia_)

    ax = sns.lineplot(x=K, y=Sum_of_squared_distances)
    ax.lines[0].set_linestyle("--")

    # Add a vertical line to show the optimum number of clusters
    plt.axvline(2, color='#F26457', linestyle=':')

    plt.xlabel('k')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Method For Optimal k')
    plt.show()

    plt.savefig('best_k.png')

def k_means_cluster(texts):
    k = 17
    # Vectorize the text
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    # Fit our Model
    model = KMeans(init="k-means++", n_clusters=k, max_iter=25, n_init=1)
    model.fit(X)

    clust_labels = model.predict(X)
    cent = model.cluster_centers_

    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()

    results_dict = {}

    for i in range(k):
        terms_list = []

        for ind in order_centroids[i, :15]:
            terms_list.append(terms[ind])

        results_dict[f'Cluster {i}'] = terms_list

    df_clusters = pd.DataFrame.from_dict(results_dict)
    print(df_clusters)


def pred_new_doc(model, vectorizer, new_docs):
    pred = model.predict(vectorizer.transform(new_docs))
    print(pred)

from wordcloud import WordCloud
from wordcloud import STOPWORDS

def word_cloud_asses(string):
    stopword_list = set(STOPWORDS)

    # Create WordCloud
    word_cloud = WordCloud(width=800, height=500,
                           background_color='white',
                           stopwords=stopword_list,
                           min_font_size=14).generate(string)

    # Set wordcloud figure size
    plt.figure(figsize=(8, 6))

    # Show image
    plt.imshow(word_cloud)

    # Remove Axis
    plt.axis("off")

    # save word cloud
    plt.savefig('wc.png')

    # show plot
    plt.show()


texts_train, texts_test = create_train_test()



import fasttext

#model = fasttext.train_unsupervised(texts_train)
print("********************************************************************************")
#print(model.predict(texts_test[100]))
print("********************************************************************************")
print("********************************************************************************")
#print(model.get_output_matrix)
print("********************************************************************************")


