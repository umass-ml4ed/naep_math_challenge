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

@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig):
    print("Config: {}".format(OmegaConf.to_yaml(cfg)))
    wandb_hyrda_cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
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
    trainer = MyTrainer(cfg, deivce=device)
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

if __name__ == '__main__':
    main()
