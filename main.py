import argparse
import random
import torch
import yaml
import json
import os
from train import MyTrainer
import utils


def add_learner_params():
    """
    The function that create args with default values
    :return: obtain the args with default values
    """
    parser = argparse.ArgumentParser(description='naep_as_challenge')
    parser.add_argument('--name', default='train', help='name for the experiment')

    #directory information
    parser.add_argument('--train_path', default='data/train.csv', help='the path of train dataset')
    parser.add_argument('--save_dir', default='data/', help='The directory to save result')
    parser.add_argument('--save_model_dir', default='saved_models/', help='The directory to save models')

    #training and testing information
    parser.add_argument('--split', action='store_true', help='no testing dataset, random split '
                                                             'training dataset as train(8/10)/val(1/10)/test(1/10)')
    parser.add_argument('--test_path', default='data/test.csv', help='the path of train dataset')

    # model definition
    parser.add_argument('--lm', default='bert-base-uncased', help='Base Language model')
    parser.add_argument('--tok', default='bert-base-uncased', help='Base Language model tokenizer')

    #0. GPTJ,
    #parser.add_argument('--task', default='all', help='train on all task')

    # task definition
    #1.base
    parser.add_argument('--base', action='store_true', help='basic classification task, '
                                                            '\ninput: response \noutput: label ')
    #2.in context
    parser.add_argument('--in_context', action='store_true', help = 'The input will include in context information')
    parser.add_argument('--closed_form', action='store_true', help = 'add closed form response to the input')
    parser.add_argument('--question_id', action='store_true', help='add question id information to the input')
    parser.add_argument('--examples', action='store_true', help= 'add examples') #random, categorywise, KNN
    parser.add_argument('--n_examples', default=1, help = 'Num of examples for each score category ')

    #label information
    parser.add_argument('--label',default=0, type=int,help = 'different type of labels: '
                                                             '\n 0: simple label without specifications (e.g. 1,2,3)'
                                                             '\n 1: detailed label with specifications (e.g. 1,1A,1B, 2A,2B,3')

    # optimizer params
    parser.add_argument('--lr_schedule', default='warmup-const')
    parser.add_argument('--opt', default='adam', help='Optimizer to use', choices=['sgd', 'adam', 'lars'])
    parser.add_argument('--iters', default=1, type=int, help='number of epochs')
    parser.add_argument('--warmup', default=0, type=float, help='number of warmup iterations in proportion to \'iters\'')
    parser.add_argument('--lr', default=2e-5, type=float, help='base learning rate')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--decay', default=0.01)

    # trainer params
    parser.add_argument('--save_freq', default=1, type=int, help='epoch frequency to save the model')
    parser.add_argument('--eval_freq', default=1, type=int, help='epoch frequency for evaluation')
    parser.add_argument('--workers', default=1, type=int, help='number of data loader workers')
    parser.add_argument('--seed', default=999, type=int, help='random seed')

    # evaluation params
    parser.add_argument('--best_metric', default='kappa', type=str, help='choose validation data')
    # extras
    parser.add_argument('--eval', action='store_true', help='only for evaluation')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    parser.add_argument('--save', action='store_true', help='save model every save_freq epochs')
    parser.add_argument('--debug', action='store_true', help='debug mode with less data')

    params = parser.parse_args()
    return params

def main(args):
    # set random seed if specified
    if args.seed != -1:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    # set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.cuda: assert device.type == 'cuda', 'no gpu found!'
    args.name = args.name + '_label' + str(args.label)
    args.root = 'logs/' + args.name + '/'
    args.save_model_dir = args.save_model_dir + args.name + '/'
    utils.safe_makedirs(args.root)
    with open(args.root+'config.yml', 'w') as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)
    print('Done with args processing, \nthe result is save to {}. '
          '\nThe model is saved to {}\nThe training data set is {}'.format(args.root,
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
    args = add_learner_params()
    main(args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
