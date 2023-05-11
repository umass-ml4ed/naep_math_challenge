import argparse
import random
import torch
import yaml
import json
import os
from train import MyTrainer
import utils
import hydra
from omegaconf import DictConfig, OmegaConf
from ExperimentLogger import ExperimentLogger as el

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
    cfg.name = cfg.name + '_label' + str(cfg.label)
    path = 'logs/' + cfg.name + '/'
    cfg.save_model_dir = cfg.save_model_dir + cfg.name + '/'
    utils.safe_makedirs(path)
    with open(path+'config.yml', 'w') as outfile:
        OmegaConf.save(cfg, outfile)
    print(f'Done with config processing, \nthe result is save to {path}. '
          f'\nThe model is saved to {cfg.save_model_dir}\nThe training data set is {cfg.train_path}')
    """
    Training
    """
    trainer = MyTrainer(cfg, device=device)
    if not cfg.eval_only:
        trainer.train()
        print('Done with training')
        """
        Evaluation
        """
        trainer.dataset_dict.pop('train')
        result = {key: trainer.evaluate(item) for key, item in list(trainer.dataset_dict.items())}
        trainer.save_best_model_and_remove_the_rest()
        trainer.save_metrics(result, 'best/')
        print('Done with evaluate')
        trainer.predict_to_save(trainer.dataset_dict['test'])

    else:
        print('No need to train')
        trainer.dataset_dict.pop('train')
        trainer.dataset_dict.pop('val')
        result = {key: trainer.evaluate(item) for key, item in list(trainer.dataset_dict.items())}
        trainer.save_best_model_and_remove_the_rest()
        trainer.save_metrics(result, '')

        test = trainer.dataset_dict['test']
        trainer.predict_to_save(test)




if __name__ == '__main__':
    main()
