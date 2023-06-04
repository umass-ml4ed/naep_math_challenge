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

def _load_data(path, sample = False, num_sample = 2000):
    with open(path, 'r') as file:
        json_load = json.load(file)
    if sample:
        json_load = random.sample(json_load, num_sample)
    else:
        if num_sample < len(json_load) and num_sample != -1:
            json_load = json_load[0:num_sample]
    x = [np.array(j['hz']) for j in json_load]
    x = np.stack(x, axis=0)
    return x, json_load

def _load_2_data(path):
    with open(path, 'r') as file:
        json_load = json.load(file)
    json_load = random.sample(json_load, 2)
    x = []
    y = []
    for j in json_load:
        x+=list(j['hz'])
        y+=list(j['gpt2'])
    x = np.array(x)
    y = np.array(y)
    return x,y

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

def boxed(file, feature='out'):
    y = []
    for f in file:
        if 'boxed' in f[feature]:
            y.append(0)
        else:
            y.append(1)
    return y

def operation(file, feature='out', o = '+'):
    y = []
    for f in file:
        if o in f[feature]:
            y.append(0)
        else:
            y.append(1)
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

def scatter_plot(X_2d,y, y_type, title = ''):
    if '/' in y_type:
        y_type = 'operation_divide'
    if '*' in y_type:
        y_type = 'operation_mul'
    if 'Let' in y_type:
        y_type = 'Define_x'
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, s=2)
    legend1 = plt.legend(*scatter.legend_elements(),
                         loc="lower left", title=y_type)

    #scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=c, s=8, marker=m, alpha=0.8, label=l)

    plt.title(y_type)
    plt.show()
    #plt.savefig('figure/'+ y_type +'_'+ title +'.png')

def scatter_plot_3d(X_3d,y,y_type):
    ax = plt.axes(projection='3d')

    scatter = ax.scatter3D(X_3d[:,0], X_3d[:,1], X_3d[:,2], c=y)
    legend1 = plt.legend(*scatter.legend_elements(),
                         loc="lower left", title=y_type)
    plt.show()


def analysis_action_feature(path):
    action_dict = {'add': [9,11,23,30,32,34,44,49,79,82,96,97,124],
                   'minus':[20,27,43,121],
                   'multiply': [14,28,38,41,45,46,53,54,99,105,148],
                   'divide':[13,67,136,139,145,150],
                   'total': [32,34,41,44,49,65,74,79,82,96,124]}
    # 'so': [77,78],
    # 'define': [15],
    # 'score': [118]}
    hz, all = _load_data(path, sample=False,num_sample=500)
    X = hz
    X_2d = TSNE(n_components=2, learning_rate='auto', init='random', random_state=0).fit_transform(X)

    boxed = [i for i, a in enumerate(all) if 'boxed' in a['out']]
    action_dict.update({'boxed': boxed})

    y = [1] * len(hz)
    action_text_dict = {}

    for i, a in enumerate(action_dict.keys()):
        print('##########', a,'############')
        temp = []
        for j in action_dict[a]:
            try:
                y[j] = 0
                temp.append(all[j]['out'])
                print(all[j]['out'])
            except:
                print('error')
        action_text_dict.update({a:temp})
        scatter_plot(X_2d, y, a)
    #print(action_text_dict)

def analysis_question(path, question_num = 1000, single=False):
    hz, all = _load_data(path, sample=False,num_sample=1000)
    X = hz
    X_2d = TSNE(n_components=2, learning_rate='auto', init='random', random_state=0).fit_transform(X)


    question_c = ''
    max_step = np.max([a['i'] for a in all])
    y = [max_step + 1] * len(hz)
    question_n = 0
    for i, a in enumerate(all):
        if question_num == question_n:
            break
        question = a['p']
        if question_c == '':
            question_c = question
        if question != question_c:
            #question_c = question_c.split('\n')[1]
            if single:
                scatter_plot(X_2d, y, question_n, title = question_c)
            question_c = question
            question_n += 1
            if single:
                y = [max_step + 1] * len(hz)

        y[i] = a['i']
    if not single:
         scatter_plot(X_2d, y, question_n, title='all')

def analysis_between_gpt2_hz(path):
    x, y = _load_2_data(path)
    label = [1]*len(x) + [0] * len(y)
    x = np.stack([x,y],axis=0)
    x = x.reshape((-1,x.shape[-1]))
    X_2d = TSNE(n_components=2, learning_rate='auto', init='random', random_state=0).fit_transform(x)
    X_3d = TSNE(n_components=3, learning_rate='auto', init='random', random_state=0).fit_transform(x)
    scatter_plot(X_2d, label, 'compare: \n 1: hz 0:gpt2')
    scatter_plot_3d(X_3d, label, 'compare: \n 1: hz 0:gpt2')

def tsne_for_different_manifold(path):
    hz, step = _load_data(path)

    n_samples = len(hz)
    n_components = 2
    (fig, subplots) = plt.subplots(3, 5, figsize=(15, 8))
    perplexities = [5, 30, 50, 100]

    X, y = datasets.make_circles(
        n_samples=n_samples, factor=0.5, noise=0.05, random_state=0
    )
    X = hz

    red = y == 0
    green = y == 1

    ax = subplots[0][0]
    ax.scatter(X[red, 0], X[red, 1], c="r")
    ax.scatter(X[green, 0], X[green, 1], c="g")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis("tight")

    for i, perplexity in enumerate(perplexities):
        ax = subplots[0][i + 1]

        t0 = time()
        tsne = manifold.TSNE(
            n_components=n_components,
            init="random",
            random_state=0,
            perplexity=perplexity,
            learning_rate="auto",
            n_iter=300,
        )
        Y = tsne.fit_transform(X)
        t1 = time()
        print("circles, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
        ax.set_title("Perplexity=%d" % perplexity)
        ax.scatter(Y[red, 0], Y[red, 1], c="r")
        ax.scatter(Y[green, 0], Y[green, 1], c="g")
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis("tight")

    # Another example using s-curve
    X, color = datasets.make_s_curve(n_samples, random_state=0)

    ax = subplots[1][0]
    ax.scatter(X[:, 0], X[:, 2], c=color)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())

    for i, perplexity in enumerate(perplexities):
        ax = subplots[1][i + 1]

        t0 = time()
        tsne = manifold.TSNE(
            n_components=n_components,
            init="random",
            random_state=0,
            perplexity=perplexity,
            learning_rate="auto",
            n_iter=300,
        )
        Y = tsne.fit_transform(X)
        t1 = time()
        print("S-curve, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))

        ax.set_title("Perplexity=%d" % perplexity)
        ax.scatter(Y[:, 0], Y[:, 1], c=color)
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis("tight")

    # Another example using a 2D uniform grid
    x = np.linspace(0, 1, int(np.sqrt(n_samples)))
    xx, yy = np.meshgrid(x, x)
    X = np.hstack(
        [
            xx.ravel().reshape(-1, 1),
            yy.ravel().reshape(-1, 1),
        ]
    )
    color = xx.ravel()
    ax = subplots[2][0]
    ax.scatter(X[:, 0], X[:, 1], c=color)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())

    for i, perplexity in enumerate(perplexities):
        ax = subplots[2][i + 1]

        t0 = time()
        tsne = manifold.TSNE(
            n_components=n_components,
            init="random",
            random_state=0,
            perplexity=perplexity,
            learning_rate="auto",
            n_iter=400,
        )
        Y = tsne.fit_transform(X)
        t1 = time()
        print("uniform grid, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))

        ax.set_title("Perplexity=%d" % perplexity)
        ax.scatter(Y[:, 0], Y[:, 1], c=color)
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.axis("tight")

    plt.show()

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

def loop_epoch_label_plots(file_name = '_loss1_multi_hz_history_train.json_history', label_type='multi'):
    epoch = [5,10,15,20, 30]
    if 'train' in file_name:
        alias = '_trainset'
    else:
        alias = '_testset'
    for e in epoch:
        path = '../analysis/result/' + str(e) + file_name
        hz, all = _load_data(path)
        X = hz
        X_2d = TSNE(n_components=2, learning_rate='auto', init='random', random_state=0).fit_transform(X)
        if label_type == 'single':
            inv_map = {v: k for k, v in label_dict.items()}
            y = [label_dict[a['label']] for a in all]
            y = np.array(y)
            multi_label_plot(X_2d, y, 'loop_single/', title='epoch_'+str(e) + alias, l_dict=inv_map)
        else:
            y_raw = [str(a['label']) for a in all]
            labels = set(y_raw)
            label_dict = {l: i for i, l in enumerate(labels)}
            inv_map = {v: k for k, v in label_dict.items()}
            y = [label_dict[t] for t in y_raw]
            multi_label_plot(X_2d, y, 'multi_single/', title='epoch_' + str(e) + alias, l_dict=inv_map)
def analyze_confusing_point(file_name = '20_loss0_hz_history_test.json_history'):
    def get_range(X_2d, index, all, rx = [0,0], ry = [0,0]):
        col0m = (X_2d[:, 0] >= rx[0]) & (X_2d[:, 0] <= rx[1])
        col1m = (X_2d[:, 1] >= ry[0]) & (X_2d[:, 1] <= ry[1])
        temp = col0m & col1m
        index = np.array(index)
        index = index[temp]
        points = [all[i] for i in index]
        return points
    #path = '../analysis/result/' + file_name
    path = '../analysis/' + file_name
    hz, all = _load_data(path)
    X = hz
    if len(X.shape) == 3:
        X = hz[:,0,:]
    X_2d = TSNE(n_components=2, learning_rate='auto', init='random', random_state=0).fit_transform(X)
    label_dict = {'one_plus': 1, 'one_mul': 2, 'one_divide': 3,
                  'one_sup': 4, 'boxed': 5, 'let': 6, 'equation': 7,
                  'mul_plus': 8, 'mul_mul': 9, 'mul_sup': 10, 'mul_divide': 11,
                  'plus_sup': 12, 'mul_mul_divide': 13, 'mixed': 14,
                  'assign': 15, 'statement': 16, 'expression': 17}
    inv_map = {v: k for k, v in label_dict.items()}
    points = get_range(X_2d, list(range(len(all))), all, rx=[-25, -15], ry=[-10, 10])

    y = [label_dict[a['label']] for a in all]
    c_x = defaultdict(list)
    for i, l in enumerate(y):
        c_x[l].append(i)


    #1. check one divide [-80,-50] [0,20]
    index = c_x[3]
    points1 = get_range(X_2d[index,:], index, all, rx=[-70, -50], ry=[0, 20])
    points2 = get_range(X_2d[index, :], index, all, rx=[30, 50], ry=[-10, 20])

    #2. check all labels in range [-30,-10], [-10,20]
    #points = get_range(X_2d, list(range(len(all))), all, rx=[-30,-10], ry=[-10,20])
    print('done')




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
    labels = df['label1'].to_list()
    label = set(list(labels))
    id2label = {}
    id_count = 0
    for elem in sorted(list(label)):
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



class Analyzer(object):
    def __init__(self, args=None, trainer:MyTrainer=None):
        self.retriever = trainer.retriever
        self.args = args
        self.trainer = trainer

    def analysis_multiple(self):
        example = self.retriever.raw
        trainer = self.trainer
        text = ['text', 'text1', 'text2']
        #text = ['text']
        model_name = 'mpnet'
        for t in text:
            data = self.retriever.create_examples_embedding(example, embed_text_name=t)
            for key, qdf in data.items():
                key = var.QUESTION_TO_NAME[key]
                analysis_label_embedding(qdf, title= key + '_'+ model_name + '_' + t)
        model_name = 'finetune_bert'
        model = trainer.model
        tok = trainer.tokenizer
        for t in text:
            data = self.retriever.create_examples_embedding(example, embed_text_name=t, model=model, tokenizer=tok, pooling='bert')
            for key, qdf in data.items():
                key = var.QUESTION_TO_NAME[key]
                analysis_label_embedding(qdf, title= key + '_'+ model_name + '_' + t)
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

    def save_small_train(self):
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


    def analysis(self):
        #self.analysis_multiple()
        #self.analysis_between_trained_not_trained()
        #self.analysis_on_similarity()
        #self.analysis_error()
        self.save_small_train()




