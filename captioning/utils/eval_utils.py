from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
from . import misc as utils

# load coco-caption if available
try:
    sys.path.append("coco-caption")
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
except:
    print('Warning: coco-caption not available')

bad_endings = ['a', 'an', 'the', 'in', 'for', 'at', 'of', 'with', 'before', 'after', 'on', 'upon', 'near', 'to', 'is',
               'are', 'am']
bad_endings += ['the']


def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0


def getCOCO(dataset, kwargs):
    if 'coco' in dataset:
        # annFile = 'coco-caption/annotations/captions_val2014.json'
        annFile = kwargs['input_coco_json']
    elif 'flickr30k' in dataset or 'f30k' in dataset:
        annFile = 'data/f30k_captions4eval.json'
    return COCO(annFile)


def language_eval(dataset, preds, preds_n, eval_kwargs, split):
    model_id = eval_kwargs['id']
    eval_oracle = eval_kwargs.get('eval_oracle', 0)

    # create output dictionary
    out = {}

    if len(preds_n) > 0:
        # vocab size and novel sentences
        if 'coco' in dataset:
            dataset_file = 'data/dataset_coco.json'
        elif 'flickr30k' in dataset or 'f30k' in dataset:
            dataset_file = 'data/dataset_flickr30k.json'
        training_sentences = set([' '.join(__['tokens']) for _ in json.load(open(dataset_file))['images'] if
                                  not _['split'] in ['val', 'test'] for __ in _['sentences']])
        generated_sentences = set([_['caption'] for _ in preds_n])
        novels = generated_sentences - training_sentences
        out['novel_sentences'] = float(len(novels)) / len(preds_n)
        tmp = [_.split() for _ in generated_sentences]
        words = []
        for _ in tmp:
            words += _
        out['vocab_size'] = len(set(words))

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    cache_path = os.path.join('eval_results/', 'cache_' + model_id + '_' + split + '.json')

    coco = getCOCO(dataset, eval_kwargs)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set
    preds_filt = [p for p in preds if p['image_id'] in valids]
    mean_perplexity = sum([_['perplexity'] for _ in preds_filt]) / len(preds_filt)
    mean_entropy = sum([_['entropy'] for _ in preds_filt]) / len(preds_filt)
    print('using {} / {} predictions'.format(len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    for metric, score in cocoEval.eval.items():
        out[metric] = score
    # Add mean perplexity
    out['perplexity'] = mean_perplexity
    out['entropy'] = mean_entropy

    imgToEval = cocoEval.imgToEval
    for k in list(imgToEval.values())[0]['SPICE'].keys():
        if k != 'All':
            out['SPICE_' + k] = np.array([v['SPICE'][k]['f'] for v in imgToEval.values()])
            out['SPICE_' + k] = (out['SPICE_' + k][out['SPICE_' + k] == out['SPICE_' + k]]).mean()
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption

    if len(preds_n) > 0:
        from . import eval_multi
        cache_path_n = os.path.join('eval_results/', 'cache_' + model_id + '_' + split + '_n.json')
        allspice = eval_multi.eval_allspice(dataset, preds_n, model_id, split)
        out.update(allspice['overall'])
        div_stats = eval_multi.eval_div_stats(dataset, preds_n, model_id, split)
        out.update(div_stats['overall'])
        if eval_oracle:
            oracle = eval_multi.eval_oracle(dataset, preds_n, model_id, split)
            out.update(oracle['overall'])
        else:
            oracle = None
        self_cider = eval_multi.eval_self_cider(dataset, preds_n, model_id, split)
        out.update(self_cider['overall'])
        with open(cache_path_n, 'w') as outfile:
            json.dump({'allspice': allspice, 'div_stats': div_stats, 'oracle': oracle, 'self_cider': self_cider},
                      outfile)

    out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
    outfile_path = os.path.join('eval_results/', 'language_' + model_id + '_' + split + '.json')
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out


def mention_precision_eval(predictions, kwargs_eval, split):
    # 评估模式：按类评价、子集绝对匹配评价、汉明距离评价
    mention_eval_mode = kwargs_eval.get('mention_eval_mode', 'class')

    outfile_path = os.path.join('eval_results/', 'mention_' + kwargs_eval['id'] + '_' + split + '.json')

    dict_output = dict()
    if mention_eval_mode == 'class':
        # 按类评价

        # 建立一个真实类别的映射表
        dict_concept_to_index = defaultdict(list)
        # 再建立一个预测类别的映射表
        dict_concept_to_index_predicition = defaultdict(list)

        for index, prediction in enumerate(predictions):
            print(prediction['list_concepts'], prediction['list_concepts_prediction'])
            for concept, concept_prediction in zip(prediction['list_concepts'], prediction['list_concepts_prediction']):
                dict_concept_to_index[concept].append(index)
                dict_concept_to_index_predicition[concept_prediction].append(index)

        dict_output['mode'] = 'class'
        dict_output['classes'] = list()
        dict_output['overall'] = dict()

        sum_p = 0.0
        sum_r = 0.0
        sum_f = 0.0
        for concept in dict_concept_to_index.keys():
            set_index = set(dict_concept_to_index[concept])
            set_index_prediction = set(dict_concept_to_index_predicition[concept])
            set_index_common = set.intersection(set_index, set_index_prediction) # 共有部分显然就是预测对的了
            print(len(set_index), len(set_index_prediction), len(set_index_common))
            p = len(set_index_common) / len(set_index_prediction) if len(set_index_prediction) > 0 else 0
            r = len(set_index_common) / len(set_index) if len(set_index) > 0 else 0
            f = (p * r * 2) / (p + r) if (p + r) > 0 else 0

            dict_output_class = {
                'concept': concept,
                'precision': p,
                'recall': r,
                'f_1': f
            }

            dict_output['classes'].append(dict_output_class)

            sum_p += p
            sum_r += r
            sum_f += f

        all_p = sum_p / len(dict_concept_to_index)
        all_r = sum_r / len(dict_concept_to_index)

        dict_output['overall'] = {
            'precision': all_p,
            'recall': all_r,
            'f_1': (2 * all_p * all_r) / (all_p + all_r)
        }

    elif mention_eval_mode == 'subset':
        # 子集匹配

        dict_output['mode'] = 'subset'
        dict_output['overall'] = dict()

        sum_IOU = 0.0
        for prediction in predictions:
            set_concept = set(prediction['list_concepts'])
            set_concept_prediction = set(prediction['list_concepts_prediction'])

            set_concept_common = set.intersection(set_concept, set_concept_prediction)
            set_concept_all = set.union(set_concept, set_concept_prediction)

            # 计算交并比
            IOU = len(set_concept_common) / len(set_concept_all)

            sum_IOU += IOU

        dict_output['overall'] = {
            'IOU': sum_IOU / len(predictions)
        }

    elif mention_eval_mode == 'subset_match':
        # 子集绝对匹配
        dict_output['mode'] = 'subset_match'
        dict_output['overall'] = dict()

        sum_match = 0.0
        for prediction in predictions:
            set_concept = set(prediction['list_concepts'])
            set_concept_prediction = set(prediction['list_concepts_prediction'])

            if set_concept == set_concept_prediction:
                sum_match += 1

        dict_output['overall'] = {
            'match': sum_match / len(predictions)
        }
    else:
        # 汉明距离，需要后面再说
        pass

    with open(outfile_path, 'w') as outfile:
        json.dump(dict_output, outfile)

    return dict_output


def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 0)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'base_val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    mention_eval = eval_kwargs.get('mention_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_n = eval_kwargs.get('sample_n', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(
        remove_bad_endings)  # Use this nasty way to make other code clean since it's a global configuration
    device = eval_kwargs.get('device', 'cuda')

    # 切换到eval模式
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    n_predictions = []  # 作者注：用于多样性评估，此时sample_n大于1

    while True:
        data = loader.get_batch(split)
        n = n + len(data['infos'])

        tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
        tmp = [_.to(device) if _ is not None else _ for _ in tmp]
        fc_feats, att_feats, labels, masks, att_masks = tmp
        if labels is not None and verbose_loss:
            # 获得当前样本的loss
            with torch.no_grad():
                loss = crit(model(fc_feats, att_feats, labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # 为每一个样本采样一个句子
        with torch.no_grad():
            tmp_eval_kwargs = eval_kwargs.copy()
            tmp_eval_kwargs.update({'sample_n': 1})
            seq, seq_logprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
            seq = seq.data
            entropy = - (F.softmax(seq_logprobs, dim=2) * seq_logprobs).sum(2).sum(1) / ((seq > 0).to(seq_logprobs).sum(1) + 1)
            perplexity = - seq_logprobs.gather(2, seq.unsqueeze(2)).squeeze(2).sum(1) / ((seq > 0).to(seq_logprobs).sum(1) + 1)

        # beam search，根据需要输出整个beam的采样结果（多样性评估使用）
        if beam_size > 1 and verbose_beam:
            for i in range(fc_feats.shape[0]):
                print('\n'.join([utils.decode_sequence(model.vocab, _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)

        sents = utils.decode_sequence(model.vocab, seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'],
                     'caption': sent,
                     'perplexity': perplexity[k].item(),
                     'entropy': entropy[k].item()}

            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']

            predictions.append(entry)

            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg'  # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image {}: {}'.format(entry['image_id'], entry['caption']))

        if sample_n > 1:
            eval_split_n(model, n_predictions, [fc_feats, att_feats, att_masks, data], eval_kwargs)

        # 回归到实际的样本测试数量（指定num_sample的话，有可能会取多）
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        else:
            num_images = ix1
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation performance... %d/%d (%f)' % (n, ix1, loss))

        if 0 <= num_images <= n:
            break

    # 多样化评估使用
    if len(n_predictions) > 0 and 'perplexity' in n_predictions[0]:
        n_predictions = sorted(n_predictions, key=lambda x: x['perplexity'])

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')

    torch.save((predictions, n_predictions), os.path.join('eval_results/', 'saved_pred_' + eval_kwargs['id'] + '_' + split + '.pth'))

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, n_predictions, eval_kwargs, split)

    if mention_eval == 1:
        split_mention = 'test'
        loader.reset_iterator(split_mention)
        num_images_mention = -1

        list_concept_novel = loader.concept_novel

        n = 0
        predictions_mention = list()

        while True:
            data = loader.get_batch(split_mention)
            n = n + len(data['infos'])

            tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [_.to(device) if _ is not None else _ for _ in tmp]
            fc_feats, att_feats, labels, masks, att_masks = tmp

            if labels is not None and verbose_loss:
                with torch.no_grad():
                    loss = crit(model(fc_feats, att_feats, labels[..., :-1], att_masks), labels[..., 1:], masks[..., 1:]).item()
                loss_sum = loss_sum + loss
                loss_evals = loss_evals + 1

            with torch.no_grad():
                tmp_eval_kwargs = eval_kwargs.copy()
                tmp_eval_kwargs.update({'sample_n': 1})
                seq, seq_logprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
                seq = seq.data
                entropy = - (F.softmax(seq_logprobs, dim=2) * seq_logprobs).sum(2).sum(1) / ((seq > 0).to(seq_logprobs).sum(1) + 1)
                perplexity = - seq_logprobs.gather(2, seq.unsqueeze(2)).squeeze(2).sum(1) / ((seq > 0).to(seq_logprobs).sum(1) + 1)

            # Print beam search
            if beam_size > 1 and verbose_beam:
                for i in range(fc_feats.shape[0]):
                    print('\n'.join([utils.decode_sequence(model.vocab, _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                    print('--' * 10)

            sents = utils.decode_sequence(model.vocab, seq)

            for k, sent in enumerate(sents):
                set_concepts_prediction = set()
                for index_token in seq[k]:
                    if index_token.item() != 0 and  model.vocab[str(index_token.item())] in list_concept_novel:
                        set_concepts_prediction.add(model.vocab[str(index_token.item())])

                entry = {'image_id': data['infos'][k]['id'],
                         'caption': sent,
                         'perplexity': perplexity[k].item(),
                         'entropy': entropy[k].item(),
                         'concept': data['concepts'][k],
                         'list_concepts': data['infos'][k]['list_concepts'],
                         'list_concepts_prediction': list(set_concepts_prediction)}

                if eval_kwargs.get('dump_path', 0) == 1:
                    entry['file_name'] = data['infos'][k]['file_path']

                predictions_mention.append(entry)

                if eval_kwargs.get('dump_images', 0) == 1:
                    cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg'  # bit gross
                    print(cmd)
                    os.system(cmd)

                if verbose:
                    print('image %s: %s' % (entry['image_id'], entry['caption']))

            # ix0 = data['bounds']['it_pos_now']
            ix1 = data['bounds']['it_max']
            if num_images_mention != -1:
                ix1 = min(ix1, num_images_mention)
            else:
                num_images_mention = ix1

            # print(ix1, num_images_mention)
            for i in range(n - ix1):
                predictions_mention.pop()

            if verbose:
                print('evaluating mention preformance... %d/%d (%f)' % (n, ix1, loss))

            if 0 <= num_images_mention <= n:
                break

        torch.save((predictions, n_predictions), os.path.join('eval_results/', 'saved_pred_' + eval_kwargs['id'] + '_' + split_mention + '.pth'))

    mention_stats = None
    if mention_eval == 1:
        mention_stats = mention_precision_eval(predictions_mention, eval_kwargs, split_mention)

    # 将模型切换回训练模式
    model.train()

    return loss_sum / loss_evals, predictions, lang_stats, mention_stats


# 多样化评估，暂时不管他
def eval_split_n(model, n_predictions, input_data, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_n = eval_kwargs.get('sample_n', 1)
    sample_n_method = eval_kwargs.get('sample_n_method', 'sample')

    fc_feats, att_feats, att_masks, data = input_data

    tmp_eval_kwargs = eval_kwargs.copy()
    if sample_n_method == 'bs':
        # case 1 sample_n == beam size
        tmp_eval_kwargs.update({'sample_n': 1, 'beam_size': sample_n, 'group_size': 1})  # randomness from softmax
        with torch.no_grad():
            model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        for k in range(fc_feats.shape[0]):
            _sents = utils.decode_sequence(model.vocab,
                                           torch.stack([model.done_beams[k][_]['seq'] for _ in range(sample_n)]))
            for sent in _sents:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                n_predictions.append(entry)
    # case 2 sample / gumbel / topk sampling/ nucleus sampling
    elif sample_n_method == 'sample' or \
            sample_n_method == 'gumbel' or \
            sample_n_method.startswith('top'):
        tmp_eval_kwargs.update(
            {'sample_n': sample_n, 'sample_method': sample_n_method, 'beam_size': 1})  # randomness from sample
        with torch.no_grad():
            _seq, _sampleLogprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        _sents = utils.decode_sequence(model.vocab, _seq)
        _perplexity = - _sampleLogprobs.gather(2, _seq.unsqueeze(2)).squeeze(2).sum(1) / (
                    (_seq > 0).to(_sampleLogprobs).sum(1) + 1)
        for k, sent in enumerate(_sents):
            entry = {'image_id': data['infos'][k // sample_n]['id'], 'caption': sent,
                     'perplexity': _perplexity[k].item()}
            n_predictions.append(entry)
    elif sample_n_method == 'dbs':
        # Use diverse beam search
        tmp_eval_kwargs.update({'beam_size': sample_n * beam_size, 'group_size': sample_n})  # randomness from softmax
        with torch.no_grad():
            model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        for k in range(loader.batch_size):
            _sents = utils.decode_sequence(model.vocab, torch.stack(
                [model.done_beams[k][_]['seq'] for _ in range(0, sample_n * beam_size, beam_size)]))
            for sent in _sents:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                n_predictions.append(entry)
    else:
        tmp_eval_kwargs.update(
            {'sample_method': sample_n_method[1:], 'group_size': sample_n, 'beam_size': 1})  # randomness from softmax
        with torch.no_grad():
            _seq, _sampleLogprobs = model(fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
        _sents = utils.decode_sequence(model.vocab, _seq)
        for k, sent in enumerate(_sents):
            entry = {'image_id': data['infos'][k // sample_n]['id'], 'caption': sent}
            n_predictions.append(entry)
    if verbose:
        for entry in sorted(n_predictions[-fc_feats.shape[0] * sample_n:], key=lambda x: x['image_id']):
            print('image %s: %s' % (entry['image_id'], entry['caption']))
