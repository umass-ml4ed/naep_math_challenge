import random
import torch
from train import MyTrainer
import utils
import hydra
from omegaconf import DictConfig, OmegaConf
from ExperimentLogger import ExperimentLogger as el
from utils.analysis import Analyzer
import os

def _construct_name(cfg):
    #the file saving name is equal to [base model]_[in context settings]_[label settings]_alias
    base = 'unk'

    if 'saved_model' in cfg.lm:
        base = cfg.lm.replace('/best','')
        base = base.replace('saved_model/','')
        if cfg.analysis or cfg.reduce:
            return base
    if cfg.reduce:
        base = ''
        return base


    if 'bert' in cfg.lm:
        base = 'bert'
    elif 't5' in cfg.lm or 'T5' in cfg.lm:
        base = 't5'
    elif 'gptj' in cfg.lm:
        base = 'gptj'
    elif 'gpt2' in cfg.lm:
        base = 'gpt2'
    elif 'alpaca' in cfg.lm:
        base = 'alpaca'
    elif 'llama' in cfg.lm:
        base = 'llama'
    if cfg.multi_head:
        base += '_multiHead'

    if cfg.non_linear_head:
        base += '_nlHead'
    if cfg.pooling == 'mean':
        base += '_meanP'
    if cfg.random :
        base += '_random'
    if cfg.same:
        base + '_same'

    if cfg.closed_form:
        base += '_c'
    if cfg.question_id:
        base += '_qid'
    if cfg.examples:
        base += '_e' + str(cfg.n_examples)
    base += '_l' + str(cfg.label)
    if cfg.task != 'all':
        base += '_' + cfg.task
    if cfg.test_fold != -1:
        base += '_fold_'+ str(cfg.test_fold) + '_' + str(cfg.val_fold)
    if cfg.prompting:
        base += '_prompting'
    base += '_loss' + str(cfg.loss)



    if len(cfg.name) != 0:
        base += '_' + cfg.name

    return base

def _sanity_check(cfg):
    assert cfg.multi_model + cfg.multi_head <= 1



@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig):
    print("Config: {}".format(OmegaConf.to_yaml(cfg)))
    wandb_hyrda_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    el.init_wandb(cfg.logging, wandb_hyrda_cfg)
    # set random seed if specified
    if cfg.seed != -1:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    if cfg.cuda: assert device.type == 'cuda', 'no gpu found!'
    if cfg.gpu_num is not None:
        cuda_str = f'cuda:{cfg.gpu_num}'
        device = torch.device(cuda_str) if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    print("Using device:", device)


    _sanity_check(cfg)
    """
    Training
    """
    if cfg.reduce:
        path = cfg.retriever.encoderModel.split('/')[:-1]
        path = '/'.join(path)
        cfg.save_model_dir = path + '/reduce_'
        if cfg.reduce_path == '':
            train_path = path + '/reduced.csv'
            all_predict_path = path + '/test_predict.csv'
        else:
            train_path = cfg.reduce_path
            all_predict_path = cfg.all_predict_path
        assert os.path.isfile(all_predict_path), 'no all prediction exists'
        cfg.all_predict_path = all_predict_path
        # no reduce have done, then process it first
        retriever_name = cfg.retriever.name
        if not os.path.isfile(train_path):
            #assert cfg.retriever.name != '', 'Please define the type of retriever'
            cfg.retriever.name = 'knn'
            cfg.name = _construct_name(cfg)
            cfg.save_model_dir = cfg.save_model_dir + cfg.name + '/'
            path = 'logs/' + cfg.name + '/'
            utils.safe_makedirs(path)
            trainer = MyTrainer(cfg, device=device)
            analyzer = Analyzer(args=cfg, trainer=trainer)
            analyzer.analysis()
            del trainer, analyzer #free gpu memory
        cfg.retriever.name = retriever_name
        assert  os.path.isfile(train_path), 'Still no reduced file found!, check setting'
        cfg.reduce_path = train_path

    if cfg.eval_only:
        cfg.name = _construct_name(cfg)
        cfg.save_model_dir = cfg.save_model_dir + cfg.name + '/'
        path = 'logs/' + cfg.name + '/'
        utils.safe_makedirs(path)
        trainer = MyTrainer(cfg, device=device)
        trainer.dataset_dict.pop('train')
        #trainer.dataset_dict.pop('val')
        test = trainer.dataset_dict['test']
        trainer.predict_to_save(test, 'test')
        trainer.predict_to_save(trainer.dataset_dict['val'], 'val')
    elif cfg.analysis:
        cfg.name = _construct_name(cfg)
        cfg.save_model_dir = cfg.save_model_dir + cfg.name + '/'
        path = 'logs/' + cfg.name + '/'
        utils.safe_makedirs(path)
        trainer = MyTrainer(cfg, device=device)
        analyzer = Analyzer(args=cfg, trainer=trainer)
        analyzer.analysis()
        print('DONE WITH LOADING ANALYZER')
        analyzer.analysis()
    elif cfg.prompting:
        cfg.name = _construct_name(cfg)
        cfg.save_model_dir = cfg.save_model_dir + cfg.name + '/'
        path = 'logs/' + cfg.name + '/'
        utils.safe_makedirs(path)
        trainer = MyTrainer(cfg, device=device)
        trainer.dataset_dict.pop('train')
        trainer.dataset_dict.pop('val')
        test = trainer.dataset_dict['test']
        trainer.prompting_predict_to_save(test, 'test')

    else:
        if cfg.task == 'all' and cfg.multi_model:
            task_list = utils.var.QUESTION_NAME
            cfg.save_model_dir = cfg.save_model_dir + _construct_name(cfg) + '/'
        else:
            task_list = [cfg.task]

        all = {}
        trainer = None
        name = cfg.name
        save_model_dir = cfg.save_model_dir
        for task in task_list:
            print('===== Train on task ', task, '================')
            cfg.task = task
            cfg.name = name
            cfg.name = _construct_name(cfg)
            print('save to ', cfg.name)
            path = 'logs/' + cfg.name + '/'
            cfg.save_model_dir = save_model_dir + cfg.name + '/'
            if 'reduce' in cfg.save_model_dir:
                cfg.save_model_dir = cfg.save_model_dir.replace('reduce_/','reduce_')
            utils.safe_makedirs(path)
            with open(path + 'config.yml', 'w') as outfile:
                OmegaConf.save(cfg, outfile)
            """
            Training
            """
            trainer = MyTrainer(cfg, device=device)
            trainer.train()
            print('Done with training')
            """
            Evaluation
            """
            trainer.dataset_dict.pop('train')
            #result = {key: trainer.evaluate(item) for key, item in list(trainer.dataset_dict.items())}
            #all.update(result)
            #trainer.save_best_model_and_remove_the_rest()
            #trainer.save_metrics(result, 'best/')
            #print('Done with evaluate')
            trainer.predict_to_save(trainer.dataset_dict['test'], 'test')
            trainer.predict_to_save(trainer.dataset_dict['val'], 'val')
        print('Save everything into ', trainer.args.output_dir)
        #trainer.save_metrics(all, 'all_')




if __name__ == '__main__':
    main()
