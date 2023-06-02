#from analysis.dataset.mathematica_with_steps import MathematicaWithStepsMathDataset
#from analysis.dataset.GSM8k import GSM8kDataset


from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
import os

def Train_SBERT_2(loss = 0, epoch = 5 , save_epoch = 5, label_type = 'multi', load_model = None):
    #file_path = 'data/word_math/train.json'
    file_path = '/home/mengxue/Dropbox/Github/ActionTuning/data/word_math/train.json'
    test_path = 'data/word_math/test.json'
    test_path = '/home/mengxue/Dropbox/Github/ActionTuning/data/word_math/test.json'
    save_name = str(save_epoch) + '_loss' + str(loss) + '_' + str(label_type)
    dataset = GSM8kDataset(dataroot=file_path, fileroot='', loss=loss, label_type=label_type)
    testset = GSM8kDataset(dataroot=test_path, fileroot='', loss=loss, label_type=label_type)
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

def Train_SBERT_3(loss = 1, epoch = 5 , save_epoch = 5, label_type = 'path', load_model = None):
    print('\nTrain on steps {} to {} epoch\n'.format(save_epoch-5, save_epoch))
    file_path = '/home/mengxue/Dropbox/Github/ALL_MATH_DATASET/gsm_8k/train_path.json'
    test_path = '/home/mengxue/Dropbox/Github/ALL_MATH_DATASET/gsm_8k/test_path.json'
    save_name = str(save_epoch) + '_loss' + str(loss) + '_' + str(label_type)
    #dataset = GSM8kDataset_path(dataroot=file_path, fileroot='', loss=loss, label_type=label_type)
    #testset = GSM8kDataset_path(dataroot=test_path, fileroot='', loss=loss, label_type=label_type)
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

def loop_train_sbert():
    epoch = [5,10,15,20,30, 35, 40,45,50,100]
    save_name = None
    for e in epoch:
        save_name = Train_SBERT_3(loss=1,epoch=5,save_epoch=e, label_type='path', load_model = save_name)


def main():
    #Train_SBERT_2()
    #test_SBERT()
    #Train_SBERT_2(loss=1, epoch=5, label_type='multi')
    #get_SBERT_embd()
    loop_train_sbert()

if __name__ == "__main__":
    main()