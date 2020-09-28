from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import captioning.utils.opts as opts
import captioning.models as models
from captioning.data.dataloader import *
from captioning.data.dataloaderraw import *
import captioning.utils.eval_utils as eval_utils
import argparse
import captioning.utils.misc as utils
import captioning.modules.losses as losses
import torch

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model', type=str, default='',
                    help='path to model to evaluate')
parser.add_argument('--cnn_model', type=str, default='resnet101',
                    help='resnet101, resnet152')
parser.add_argument('--infos_path', type=str, default='',
                    help='path to infos to evaluate')
parser.add_argument('--only_lang_eval', type=int, default=0,
                    help='lang eval on saved results')
parser.add_argument('--only_mention_eval', type=int, default=0,
                    help='mention eval on saved results')
parser.add_argument('--force', type=int, default=0,
                    help='force to evaluate no matter if there are results available')
parser.add_argument('--device', type=str, default='cuda',
                    help='cpu or cuda')

opts.add_eval_options(parser)
opts.add_diversity_opts(parser)
opt = parser.parse_args()

print('loading information from {}'.format(opt.infos_path))
with open(opt.infos_path, 'rb') as f:
    infos = utils.pickle_load(f)

# 需要将加载到的参数中一些必要的修改项（数据集在位置）进行调整
replace = ['input_fc_dir', 'input_att_dir', 'input_box_dir', 'input_label_h5', 'input_json', 'batch_size', 'id']
ignore = ['start_from']

for k in vars(infos['opt']).keys():
    if k in replace:
        setattr(opt, k, getattr(opt, k) or getattr(infos['opt'], k, ''))
    elif k not in ignore:
        if not k in vars(opt):
            vars(opt).update({k: vars(infos['opt'])[k]})  # copy over options from model

vocab = infos['vocab']  # ix -> word mapping

pred_fn = os.path.join('eval_results/', 'saved_pred_' + opt.id + '_' + opt.split + '.pth')
language_fn = os.path.join('eval_results/', 'language_' + opt.id + '_' + opt.split + '.json')
mention_fn = os.path.join('eval_results/', 'mention_' + opt.id + '_' + opt.split + '.json')

if opt.only_lang_eval == 1 or (not opt.force and os.path.isfile(pred_fn)):
    # 预测结果已经存在，只需要考虑是否重新评估语言性能
    if not opt.force:
        try:
            if os.path.isfile(language_fn):
                print(language_fn)
                json.load(open(language_fn, 'r'))
                print('already evaluated')
                os._exit(0)
        except:
            pass

    predictions, n_predictions = torch.load(pred_fn)
    lang_stats = eval_utils.language_eval(opt.input_json, predictions, n_predictions, vars(opt), opt.split)
    print(lang_stats)
    os._exit(0)

if opt.only_mention_eval == 1 or (not opt.force and os.path.isfile(pred_fn)):
    # 预测结果已经存在，只需要考虑是否重新评估语言性能
    if not opt.force:
        try:
            if os.path.isfile(mention_fn):
                print(mention_fn)
                json.load(open(mention_fn, 'r'))
                print('already evaluated')
                os._exit(0)
        except:
            pass

    predictions, _ = torch.load(pred_fn)
    mention_stats = eval_utils.mention_precision_eval(predictions, vars(opt), opt.split)
    print(mention_stats)
    os._exit(0)

if not opt.force:
    # 非强制，如果已经有了评估结果，不再重新处理
    try:
        tmp = torch.load(pred_fn)
        if opt.language_eval == 1:
            json.load(open(language_fn, 'r'))
        if opt.mention_eval == 1:
            json.load(open(mention_fn, 'r'))
        print('Result is already there')
        os._exit(0)
    except:
        pass

# 默认模式，从头评估一遍
opt.vocab = vocab
model = models.setup(opt)
del opt.vocab

model.load_state_dict(torch.load(opt.model, map_location='cpu'))
model.to(opt.device)
model.eval()

crit = losses.LanguageModelCriterion()

# # Create the Data Loader instance
# if len(opt.image_folder) == 0:
loader = DataLoader(opt)
# else:
#     loader = DataLoaderRaw({'folder_path': opt.image_folder,
#                             'coco_json': opt.coco_json,
#                             'batch_size': opt.batch_size,
#                             'cnn_model': opt.cnn_model})

# 作者注：使用预训练模型时，可能其词汇表和我们处理编码的json文件中的caption不一致，因此要替换成模型对应的词汇表
loader.dataset.ix_to_word = infos['vocab']

opt.dataset = opt.input_json
loss, split_predictions, lang_stats, mention_stats = eval_utils.eval_split(model, crit, loader, vars(opt))

print('loss:', loss)
if lang_stats:
    print('language statistics:')
    print(lang_stats)

if mention_stats:
    print('mention statistics:')
    print(mention_stats)

if opt.dump_json == 1:
    json.dump(split_predictions, open('vis/vis.json', 'w'))
