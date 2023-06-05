from sklearn.manifold import TSNE
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from time import time
from langdetect import detect, detect_langs
import numpy as np
from scipy import linalg
import matplotlib as mpl
import random
import  json
from sklearn.cluster import KMeans
import numpy as np
from model.ModelFactory import ModelFactory as mf
from datasets import load_metric
from collections import defaultdict
from sklearn.cluster import DBSCAN, OPTICS
from transformers import XLNetTokenizer
import itertools
from sklearn import mixture
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from model.examplesRetriever import KNNRetriever
from tqdm import tqdm
from utils import var
from train import MyTrainer
from train import IncontextDataset, prepare_dataset
import pandas as pd
import os
import copy
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics


random.seed(0)
#color = ['rosybrown','lightcoral',
#         'maroon','tomato','lightsalmon','chocolate',
#         'tan','orange','gold',
#         'yellowgreen', 'greenyellow', 'forestgreen', 'paleturquoise',
#         'teal', 'deepskyblue', 'lightskyblue','lavender', 'thistle', 'hotpink']
color = ['maroon','gold','limegreen', 'deepskyblue', 'red', 'hotpink', 'green', 'black']
GREY = 'lightgray'
color_iter = itertools.cycle(color)
markers = ['.','o','v', '8','p','*','x','d']
marker_iter = itertools.cycle(markers)



def langua_detect(file, feature='out'):
    y = []
    for f in file:
      text = f[feature]
      try:
        ls =  detect_langs(text)
        en = 0
        for l in ls:
          if l.lang == 'en':
              y.append(l.prob)
              en += 1
        if en == 0:
            #rest_prob = [ for l ]
            y.append(0)
      except:
        y.append(0)

    assert len(y) == len(file)
    return y

def Gaussian_clustering(X, X_2d, n_components=20):
    def _plot_results(X, Y_, means, covariances, index, title):
        splot = plt.subplot(2, 1, 1 + index)
        for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
            v, w = linalg.eigh(covar)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            # as the DP will not use every component it has access to
            # unless it needs it, we shouldn't plot the redundant
            # components.
            if not np.any(Y_ == i):
                continue
            plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], 0.8, color=color)

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180.0 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)

        #plt.xlim(-40, 40)
        #plt.ylim(-20, 20)
        plt.xticks(())
        plt.yticks(())
        plt.title(title)
    np.random.seed(0)
    # Fit a Gaussian mixture with EM using five components
    gmm = mixture.GaussianMixture(n_components=n_components, covariance_type="full").fit(X)
    _plot_results(X_2d, gmm.predict(X), gmm.means_, gmm.covariances_, 0, "Gaussian Mixture")

    # Fit a Dirichlet process Gaussian mixture using five components
    dpgmm = mixture.BayesianGaussianMixture(n_components=n_components, covariance_type="full").fit(X)
    _plot_results(
        X_2d,
        dpgmm.predict(X),
        dpgmm.means_,
        dpgmm.covariances_,
        1,
        "Bayesian Gaussian Mixture with a Dirichlet process prior",
    )

    plt.savefig('figure/Gaussian.png')
    #plt.show()

    return dpgmm.predict(X)

def dbscan_clustering(X,X_2d = None, path=None):
    if X_2d is None:
        X_2d = TSNE(n_components=2, learning_rate='auto', init='random', random_state=0).fit_transform(X)
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X_2d[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = X_2d[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )
    plt.title(f"Estimated number of clusters: {n_clusters_}")
    if path is None:
        plt.show()
    else:
        plt.savefig(path + '.png')
    plt.close()

def kmean_clustering(X, X_2d = None, path=None):
    if X_2d is None:
        X_2d = TSNE(n_components=2, learning_rate='auto', init='random', random_state=0).fit_transform(X)
    km = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
    labels = km.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[km.core_sample_indices_] = True

    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = labels == k

        xy = X_2d[class_member_mask & core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=14,
        )

        xy = X_2d[class_member_mask & ~core_samples_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6,
        )
    plt.title(f"Estimated number of clusters: {n_clusters_}")
    if path is None:
        plt.show()
    else:
        plt.savefig(path + '.png')
    plt.close()


def analysis_cluster(data, y, text='in'):
    #nlp = spacy.load("en_core_web_trf")
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    from collections import Counter
    #nlp = spacy.load("en_core_web_sm")
    y_labels = set(y)
    clusters = defaultdict(list)
    for i, y0 in enumerate(y):
        clusters[y0].append(data[i])
    #1. key word extraction
    for y in y_labels:
        all_words = [ a[text] for a in clusters[y]]
        all_words = ' '.join(all_words)
        #result = nlp(all_words)
        words = tokenizer.tokenize(all_words)
        #words = [token.text for token in result if token.is_stop != True and token.is_punct != True]
        word_freq = Counter(words)
        common_words = word_freq.most_common(50)
        print(common_words)
        #print('Label {} result: '.format(y), result.ents)

    json_cluster = {}
    for key in clusters.keys():
        temp = clusters[key]
        temp = [t['in'] for t in temp]
        json_cluster[int(key)] = temp
    with open('result/clustering_score_gaussion.json', 'w') as outfile:
        json.dump(json_cluster, outfile, indent=4)

def multi_label_plot(X_2d, y, y_type, title= '', n_per_fig = 5, l_dict = None, top_n = 2):
    import math

    c_x = defaultdict(list)
    for i, l in enumerate(y):
        c_x[l].append(i)
    c_x = dict(sorted(c_x.items(), key=lambda x: len(x[1]), reverse=True))
    labels = list(c_x.keys())
    n_figure = math.ceil(len(labels) / n_per_fig)
    start = 0
    end = start + n_per_fig
    for n in range(n_figure):
        if n >= top_n:
            break
        plt.figure()
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=mcolors.CSS4_COLORS[GREY],
                              s=8, marker=markers[0], alpha=0.8, label = 'all')
        if end <= len(labels):
            labels_n = labels[start:end]
            start = end
            end += n_per_fig
        else:
            assert n == n_figure - 1
            labels_n = labels[start:]
        for l, c, m in zip(labels_n, color, markers):
            index = c_x[l]
            c = mcolors.CSS4_COLORS[c]
            l = l_dict[l]
            scatter = plt.scatter(X_2d[index, 0], X_2d[index, 1], c=c, s=8, marker=m, alpha=0.8, label = l)
            #legend1 = plt.legend(*scatter.legend_elements(),
            #                     loc="lower left", title=y_type)
            plt.title(title)
        plt.legend()
        plt.savefig('figure/' + y_type + title + str(n) + '.png')

def plot_with_embed_label(embed, label):
    X_2d = TSNE(n_components=2, learning_rate='auto', init='random', random_state=0).fit_transform(embed)

    fig, ax = plt.subplots()
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], s=2)
    for i, txt in enumerate(label):
        ax.annotate(txt, (X_2d[i,0], X_2d[i,1]))
    fig.show()


def analysis_label_embedding(df, key='key', title='title', path = None ):
    if len(df) > 3000:
        df = df.sample(n=3000)

    embed = np.array(df['emb'].to_list())
    labels = df['label1'].astype(str).to_list()
    if key == 'all':
        labels = df['qid'].to_list()


    label = set(list(labels))
    id2label = {}
    id_count = 0
    for elem in list(label):
        id2label[id_count] = elem
        id_count += 1
    label2id = dict((v, k) for k, v in id2label.items())
    y = [label2id[i] for i in labels]
    X = embed
    X_2d = TSNE(n_components=2, learning_rate='auto', init='random', random_state=0).fit_transform(X)
    c_x = defaultdict(list)
    for i, l in enumerate(y):
        c_x[l].append(i)
    for l, c, m in zip(id2label.keys(), color, markers):
        index = c_x[l]
        c = mcolors.CSS4_COLORS[c]
        l_name = id2label[l]
        scatter = plt.scatter(X_2d[index, 0], X_2d[index, 1], c=c, s=8, marker=m, alpha=0.8, label=l_name)
    plt.legend(scatterpoints=3, title=key)
    #plt.show()
    if path is None:
        plt.show()
    else:
        plt.savefig(path + '.png')
    plt.close()

    #analysis clustering
    # if path is None:
    #     dbscan_clustering(X, X_2d)
    # else:
    #     dbscan_clustering(X, X_2d, path + 'clustering')



class Analyzer(object):
    def __init__(self, args=None, trainer:MyTrainer=None):
        self.retriever = trainer.retriever
        self.args = args
        self.trainer = trainer

    def analysis_multiple(self):
        example = self.retriever.raw
        trainer = self.trainer
        text = ['text']
        #text = ['text']
        # model_name = 'knn'
        # for t in text:
        #     data = self.retriever.create_examples_embedding(example, embed_text_name=t)
        #     for key, qdf in data.items():
        #         key = var.QUESTION_TO_NAME[key]
        #         analysis_label_embedding(qdf, title= key + '_'+ model_name + '_' + t)

        model_name = 'finetune_bert'
        model = trainer.model
        tok = trainer.tokenizer
        for t in text:
            data = self.retriever.create_examples_embedding(example, embed_text_name=t, model=model, tokenizer=tok, pooling='bert')
            analysis_label_embedding(pd.concat(list(data.values())), key='all')
            for key, qdf in data.items():
                key = var.QUESTION_TO_NAME[key]
                analysis_label_embedding(qdf, key= key)
    def analysis_between_trained_not_trained(self):
        example = self.retriever.raw
        trainer = self.trainer
        t = 'text'

        data1 = self.retriever.create_examples_embedding(example, embed_text_name=t)
        model = trainer.model
        tok = trainer.tokenizer
        data2 = self.retriever.create_examples_embedding(example, embed_text_name=t, model=model, tokenizer=tok,
                                                        pooling='bert')

        model_name = 'compare_bert'
        for key in data1.keys():
            qdf1 = data1[key]
            qdf2 = data2[key]
            qdf1['label1'] = 'o_' + qdf1['label1']
            qdf = pd.concat([qdf1, qdf2])
            key = var.QUESTION_TO_NAME[key]
            analysis_label_embedding(qdf, title=key + '_' + model_name + '_' + t)
            #qdf3 = qdf[qdf['label1'].isin(['o_1A','o_1B','1A','1B'])]
            #qdf3 = qdf[qdf['label1'].isin(['o_2A', 'o_2B', '2A', '2B', '3', 'o_3'])]
    def analysis_on_similarity(self):
        trainer = self.trainer
        t = 'text'
        data1 = self.retriever.obtain_embedding_as_df()
        model = trainer.model
        tok = trainer.tokenizer
        data = self.retriever.raw
        for key in data1.keys():
            # only obtain closed one
            qdf1 = data1[key]
            qdf1 = qdf1[qdf1['label1'].isin(['3', '2A', '2B'])]
            example = []
            percentage = []
            for d in tqdm(qdf1.iterrows(), total=len(qdf1), position=0):
                d = d[1]
                e = self.retriever.fetch_examples(d)
                percentage.append(e['label1'].value_counts().to_dict())
                example.append(self.retriever.fetch_examples(d))
            qdf1['example_label'] = percentage
            example = pd.concat(example)
            example = pd.concat([example, qdf1])
            example_ids = example['id'].to_list()

            qdf2 = self.retriever.create_examples_embedding(data[data['qid']==key], embed_text_name=t, model=model, tokenizer=tok,
                                                             pooling='bert')
            qdf2 = qdf2[key]
            qdf_left = qdf2[qdf2['id'].isin(example_ids)]
            percentage = []
            for d in tqdm(qdf1.iterrows(), total=len(qdf1), position=0):
                d = d[1]
                e = self.retriever.fetch_examples(d, model=model, tok=tok, pooling='bert')
                percentage.append(e['label1'].value_counts().to_dict())
            qdf1['trained_example_label'] = percentage
            analysis_label_embedding(qdf_left, title=key + '_' + 'similar')
    print('done')

    def analysis_error(self):
        path = self.trainer.input_args.lm
        t='text'
        trainer = self.trainer
        model = trainer.model
        tok = trainer.tokenizer
        if 'best' in path:
            path = path.replace('/best', '')
        path = path + '/test_predict.csv'
        test = prepare_dataset(pd.read_csv(path), self.trainer.input_args)
        train = self.trainer.train_dataset
        test_dataset = IncontextDataset(tokenizer=self.trainer.tokenizer, data=test, args=self.trainer.input_args,
                                        labels_dict=self.trainer.label2id, example=train,
                                        question_dict=self.trainer.question2id, retriever=self.retriever, eval=True)
        test = test_dataset.to_pandas()
        test_df = self.retriever.create_examples_embedding(test, embed_text_name=t, model=model, tokenizer=tok, pooling='bert')
        analysis_label_embedding(test_df, title= '_' + 'test')

        print('done')


    def save_top_k(self):
        pass

    def save_small_train_from(self):
        path = self.trainer.input_args.lm
        path = path.replace('/best', '')
        path = path #+ '/reduced_train.csv'
        def save_list(df):
            name = ['train', 'val','test']
            reduced = {}
            for key in name:
                df_train = df[df['top_k_'+ key].notna()]
                train_list = df_train['top_k_' + key].apply(eval).tolist()
                flattened_list = [item for sublist in train_list for item in sublist]
                reduce = list(set(flattened_list))
                reduced[key] = reduce
                if key == 'train':
                    reduced[key] = reduce + df_train['id'].tolist()
            with open(path + '/reduced_list.json', "w") as file:
                json.dump(reduced, file)
        train_0 = self.trainer.train_dataset
        train_0 = train_0.to_pandas()
        def reduce_data(data, train_0, text='train'):
            train = data
            labels = ['2', '2A', '2B', '3']
            labels_1 = ['1', '1A', '1B']
            train = train[train['label1'].isin(labels)]
            ids = []
            id_item = []
            for d in tqdm(train.iterrows(), total=len(train), position=0):
                d = d[1]
                e = self.retriever.fetch_examples(d)
                if text == 'train':
                    e = e[e['label1'].isin(labels_1)]
                # reduced_examples.append(e)
                ids += e['id'].tolist()
                id_item.append(ids)
            ids = set(ids)
            train['top_k_' + text] = id_item
            reduced_examples = train_0[train_0['id'].isin(ids)]
            # reduced_examples['label1']='reduced'
            # check = pd.concat([train_0, reduced_examples])
            # test_df = self.retriever.create_examples_embedding(check)
            # test_df = list(test_df.values())[0]
            # analysis_label_embedding(test_df, title= '_' + 'reduced')
            return reduced_examples, train

        if os.path.isfile(path + '/reduced_train.csv'):
            df = pd.read_csv(path + '/reduced_train.csv')
        else:
            train_reduced, train_with_id = reduce_data(train_0, train_0)
            train_reduced.to_csv(path + '/r_train.csv', index=False)
            print('save train')

            test_0 = self.trainer.test_dataset
            test_0 = test_0.to_pandas()
            self.retriever.create_examples_embedding(test_0, test=True)
            test_reduced, test_with_id = reduce_data(train_with_id, test_0, text='test')
            #test_reduced.to_csv(path + '/r_test.csv', index=False)
            #print('save test')

            val_0 = self.trainer.eval_dataset
            val_0 = val_0.to_pandas()
            self.retriever.create_examples_embedding(val_0, test=True)
            val_reduced, val_with_id = reduce_data(test_with_id, val_0, text='val')
            val_reduced.to_csv(path + '/r_val.csv', index=False)
            print('save val')

            val_with_id.to_csv(path + '/r_with_id.csv', index=False)
            print('save id file', len(val_with_id))

            # train = train_0[train_0['label1'].isin(labels)]
            print('train_reduced with # ', len(train_reduced))
            reduced_examples = pd.concat([val_with_id, train_reduced, test_reduced, val_reduced])
            reduced_examples['score_to_predict'] = reduced_examples['label']
            reduced_examples['label'] = reduced_examples['label1']
            reduced_examples.to_csv(path + '/reduced_train.csv', index=False)
            df = val_with_id
        save_list(df)

    def select_subgroup(self, path='test_predict.csv'):
        pass



    def analysis(self):
        #self.analysis_multiple()
        #self.analysis_between_trained_not_trained()
        #self.analysis_on_similarity()
        #self.analysis_error()
        self.save_small_train()




