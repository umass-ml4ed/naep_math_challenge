import random
import torch
from train import MyTrainer
import utils
import hydra
from omegaconf import DictConfig, OmegaConf
from ExperimentLogger import ExperimentLogger as el
from model.ModelFactory import ModelFactory as mf
from utils.ResponseParser import ResponseParser as rp

def _construct_name(cfg):
    #the file saving name is equal to [base model]_[in context settings]_[label settings]_alias
    base = 'unk'
    if 'bert' in cfg.lm:
        base = 'bert'
    elif 't5' or 'T5' in cfg.lm:
        base = 't5'
    elif 'gptj' in cfg.lm:
        base = 'gptj'

    if cfg.multi_head:
        base += '_multiHead'

    if cfg.closed_form:
        base += '_c'
    if cfg.question_id:
        base += '_qid'
    if cfg.examples:
        base += '_e' + str(cfg.n_examples)
    base += '_l' + str(cfg.label)
    if cfg.task != 'all':
        base += '_' + cfg.task
    #if cfg.seed != -1:
    #    base += '_seed' + str(cfg.seed)
    if cfg.test_fold != -1:
        base += '_fold_'+ str(cfg.test_fold) + '_' + str(cfg.val_fold)
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
    if cfg.gpu_num is not None:
        cuda_str = f'cuda:{cfg.gpu_num}'
        device = torch.device(cuda_str) 

    if cfg.cuda: assert device.type == 'cuda', 'no gpu found!'

    _sanity_check(cfg)

    #print(f'Done with config processing, \nthe result is save to {path}. '
    #      f'\nThe model is saved to {cfg.save_model_dir}\nThe training data set is {cfg.train_path}')
    """
    Training
    """


    if cfg.eval_only:
        trainer = MyTrainer(cfg, device=device)
        trainer.dataset_dict.pop('train')
        trainer.dataset_dict.pop('val')
        #result = {key: trainer.evaluate(item) for key, item in list(trainer.dataset_dict.items())}
        #trainer.save_metrics(result, '')
        test = trainer.dataset_dict['test']
        trainer.predict_to_save(test, 'test_')
    elif cfg.prompt_model:
        prompts = ["Hi", "Tell me a joke"] # TODO populate me with actual data
        (model, tokenizer) = mf.produce_model_and_tokenizer(cfg, None, None, None)
        print("Sending model to device this might take a while.")
        model.to(device)

        # TODO this loop should probably live in another file
        # TODO batching the below op
        completions = []
        predictions = []
        for prompt in prompts:
            inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
            input_length = inputs.size()[1]
            output = model.generate(inputs, max_length=input_length + 50, num_return_sequences=1, do_sample=True)
            completion = tokenizer.decode(output[:, input_length:][0], skip_special_tokens=True)
            completions.append(completion)
            prediction = rp.parse_label(completion)
            predictions.append(prediction)

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
            result = {key: trainer.evaluate(item) for key, item in list(trainer.dataset_dict.items())}
            all.update(result)
            trainer.save_best_model_and_remove_the_rest()
            trainer.save_metrics(result, 'best/')
            print('Done with evaluate')
            trainer.predict_to_save(trainer.dataset_dict['test'])
        print('Save everything into ', trainer.args.output_dir)
        trainer.save_metrics(all, 'all_')




if __name__ == '__main__':
    main()
