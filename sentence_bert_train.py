#from analysis.dataset.mathematica_with_steps import MathematicaWithStepsMathDataset
#from analysis.dataset.GSM8k import GSM8kDataset

from model.dataset import SentenceBertDataset
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, losses
from model.SentenceTransformer import SentenceTransformer
import os
from omegaconf import DictConfig, OmegaConf
from train import pd, split_data_into_TrainValTest,var ,rerange_data,rerange_examples, KNNRetriever, prepare_dataset
import hydra

def Train_SBERT_2(loss = 0, epoch = 5 , save_epoch = 5, label_type = 'multi', load_model = None):
    #file_path = 'data/word_math/train.json'
    file_path = '/home/mengxue/Dropbox/Github/ActionTuning/data/word_math/train.json'
    test_path = 'data/word_math/test.json'
    test_path = '/home/mengxue/Dropbox/Github/ActionTuning/data/word_math/test.json'
    save_name = str(save_epoch) + '_loss' + str(loss) + '_' + str(label_type)
    #dataset = GSM8kDataset(dataroot=file_path, fileroot='', loss=loss, label_type=label_type)
    #testset = GSM8kDataset(dataroot=test_path, fileroot='', loss=loss, label_type=label_type)
    dataset = testset = None
    data_batch = DataLoader(dataset, shuffle=False, batch_size=16)
    if load_model == None:
        model = SentenceTransformer('bert-base-nli-mean-tokens')
    else:
        model = SentenceTransformer(os.path.abspath('analysis/models/' + load_model))
    if loss == 2:
        train_loss = losses.BatchHardTripletLoss(model = model)
    elif loss == 0:
        train_loss = losses.ContrastiveLoss(model=model)
    else:
        train_loss = losses.CosineSimilarityLoss(model=model)
    model.fit(train_objectives = [(data_batch, train_loss)],
              epochs=epoch, warmup_steps=100,
              output_path = 'analysis/models/' + save_name)
    dataset.sbert_encode(type='history', file_name='result/' + save_name + '_hz_history_train.json', model=model)
    testset.sbert_encode(type='history', file_name='result/' + save_name + '_hz_history_test.json', model=model)
    return save_name

def Train_SBERT_3(loss = 1, epoch = 5 , save_epoch = 5, load_model = None, args=None):
    print('\nTrain on steps {} to {} epoch\n'.format(save_epoch-5, save_epoch))
    file_path = 'data/train.csv'
    save_name = str(save_epoch) + '_loss' + str(loss) #+ '_' + str(label_type)
    dataset = 0
    testset = 0
    data_batch = DataLoader(dataset, shuffle=False, batch_size=16)
    if load_model == None:
        model = SentenceTransformer('bert-base-nli-mean-tokens')
    else:
        model = SentenceTransformer(os.path.abspath('analysis/models/' + load_model))
    if loss == 2:
        train_loss = losses.BatchHardTripletLoss(model = model)
    elif loss == 0:
        train_loss = losses.ContrastiveLoss(model=model)
    else:
        train_loss = losses.CosineSimilarityLoss(model=model)
    if save_epoch == 5:
        dataset.evaluation(model=model, epoch=0)
        testset.evaluation(model=model, epoch=0)
    model.fit(train_objectives = [(data_batch, train_loss)],
              epochs=epoch, warmup_steps=100,
              output_path = 'analysis/models/' + save_name, show_progress_bar=False)
    dataset.sbert_encode(type='path', file_name='result/' + save_name + '_hz_history_train.json', model=model)
    testset.sbert_encode(type='path', file_name='result/' + save_name + '_hz_history_test.json', model=model)
    dataset.evaluation(model=model, epoch=save_epoch)
    testset.evaluation(model=model, epoch=save_epoch)

    return save_name
def test_SBERT(loss = 0, load_name = '20_loss0'):
    file_path = 'data/word_math/train.json'
    test_path = 'data/word_math/test.json'
    #dataset = GSM8kDataset(dataroot=file_path, fileroot='', loss=loss)
    #testset = GSM8kDataset(dataroot=test_path, fileroot='', loss=loss)
    dataset = 0
    testset = 0
    model = SentenceTransformer(os.path.abspath('analysis/models/'+ load_name))
    dataset.sbert_encode(type='history', file_name='result/' + load_name + '_hz_history_train.json', model=model)
    testset.sbert_encode(type='history', file_name='result/' + load_name +'_hz_history_test.json', model=model)

def get_SBERT_embd(loss=0):
    model0 = 'bert-base-nli-mean-tokens'
    load_names = ['5_loss0', '10_loss0', '15_loss0', '20_loss0']
    abs_load_names = [os.path.abspath('analysis/models/'+l) for l in load_names]
    abs_load_names = [model0] + abs_load_names
    file_path = 'data/word_math/train.json'
    test_path = 'data/word_math/test.json'
    #dataset = GSM8kDataset(dataroot=file_path, fileroot='', loss=loss)
    #testset = GSM8kDataset(dataroot=test_path, fileroot='', loss=loss)
    dataset = 0
    testset = 0
    for n1,n2 in zip(['0_loss0'] + load_names, abs_load_names):
        model = SentenceTransformer(n2)
        #dataset.sbert_encode(type='history', file_name='result/' + n1 + '_hz_history_train.json', model=model)
        testset.sbert_encode(type='history', file_name='result/' + n1 + '_hz_history_test.json', model=model)

def loop_train_sbert(cfg):
    epoch = [5,10,15,20,30, 35, 40,45,50,100]
    save_name = None
    for e in epoch:
        save_name = Train_SBERT_3(loss=1,epoch=5,save_epoch=e, label_type='path', load_model = save_name, args=cfg)

def run(args):
    training_dataset = pd.read_csv(args.train_path)
    training_dataset = prepare_dataset(training_dataset, args)



    if args.task != 'all':
        if args.task not in var.QUESTION_LIST:
            args.task = var.NAME_TO_QUESTION[args.task]
        training_dataset = training_dataset[training_dataset['qid'] == args.task]
    training_dataset['label'] = training_dataset['label'].astype(str)
    labels = set(list(training_dataset['label']))
    id2label = {}
    id_count = 0
    for elem in sorted(list(labels)):
        id2label[id_count] = elem
        id_count += 1
    label2id = dict((v, k) for k, v in id2label.items())
    num_label = len(labels)

    if args.reduce:
        train, val, test = split_data_into_TrainValTest(training_dataset, args=args)
        iddf = pd.read_csv(args.reduce_path)
        reduce_list = iddf['id'].tolist()
        train = train[train['id'].isin(reduce_list)]
        val = val[val['id'].isin(reduce_list)]
        test = test[test['id'].isin(reduce_list)]
    else:
        train, val, test = split_data_into_TrainValTest(training_dataset, args=args)
    rerange_data(train, args)
    rerange_data(val, args)
    rerange_data(test, args)
    _, examples = rerange_examples(train)
    model = SentenceTransformer(args.lm)
    #model = SentenceTransformer('saved_models/sbert_VH271613/10')
    model = model.cuda()
    tokenizer = model.tokenizer


    retriever = KNNRetriever(args, num_label=num_label, id2label=id2label, label2id=label2id,
                             model=model, tokenizer=tokenizer, pooling='sbert', model_str='sbert')
    if args.debug:
        train = train.sample(n=1000)
    retriever.create_examples_embedding(train)
    #retriever.create_examples_embedding(test)

    if args.retriever.name == 'knn':
        retriever.create_topk_list_for_each_item()
        dataset = SentenceBertDataset(tokenizer=tokenizer, data=train, args=args,
                                     labels_dict=label2id, question_dict=None, retriever=retriever)
    else:
        dataset = SentenceBertDataset(tokenizer=tokenizer, data=train, args=args,
                                     labels_dict=label2id, question_dict=None)

    data_batch = DataLoader(dataset, shuffle=False, batch_size=args.batch_size)
    if args.loss == 2:
        train_loss = losses.BatchHardTripletLoss(model = model)
    elif args.loss == 0:
        train_loss = losses.ContrastiveLoss(model=model)
    else:
        train_loss = losses.CosineSimilarityLoss(model=model)

    output_path = 'saved_models/sbert_' + args.task
    model.fit(train_objectives = [(data_batch, train_loss)],
              epochs=args.iters, warmup_steps=100,
              output_path = output_path, show_progress_bar=True,
              checkpoint_path=output_path, checkpoint_save_total_limit=5, checkpoint_save_steps = 5,
              retriever=retriever, args = args)

@hydra.main(version_base=None, config_path="conf", config_name="sbert")
def main(cfg: DictConfig):
    #Train_SBERT_2()
    #test_SBERT()
    #Train_SBERT_2(loss=1, epoch=5, label_type='multi')
    #get_SBERT_embd()
    #loop_train_sbert(cfg)
    run(cfg)

if __name__ == "__main__":
    main()