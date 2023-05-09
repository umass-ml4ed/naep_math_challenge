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
    args = cfg
    # set random seed if specified
    if args.seed != -1:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    if args.cuda: assert device.type == 'cuda', 'no gpu found!'
    args.name = args.name + '_label' + str(args.label)
    path = 'logs/' + args.name + '/'
    args.save_model_dir = args.save_model_dir + args.name + '/'
    utils.safe_makedirs(path)
    with open(path+'config.yml', 'w') as outfile:
        OmegaConf.save(cfg, outfile)
    print('Done with args processing, \nthe result is save to {}. '
          '\nThe model is saved to {}\nThe training data set is {}'.format(path,
                                                                            args.save_model_dir, args.train_path))
    """
    Training
    """
    trainer = MyTrainer(args, deivce=device)
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
    # args = add_learner_params()
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
