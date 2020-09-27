"""
    caption编码和处理实现
    原始处理：
        1. 读取Karpathy的配置进行编码处理，图像的路径信息和划分信息保存在json中，caption编码信息保存hdf5文件中
        2. 由于已经进行了分词，这里的caption欲处理只过滤极低频词，替换为UNK
        3. 对caption的单词索引信息也保存在json文件中，其中0表示空或者EOS，所有实际单词包括UNK
        4. caption的编码结果使用数组存储，具体长度使用额外的一个数组保存
    新增内容：
        1. 从NBT的dic_coco.json中获取wtod（concept词表）和wtol（lemma词表）两个词典，用于后续处理
        2. 指令部分根据输入和输出的修改进行了调整
        3. 将随机种子作为参数可以选择（因为不同的实现用的随机种子不一样）
        4. 构建词汇表、lemma表、concept表等映射表
        5. 根据way参数，随机抽取指定个数的concept作为novel concepts
        6. 提取所有image包含的caption中出现的concept情况，记录每个image样本的concept表（目前区分单复数，不记录出现次数和位置）
        7. 除了词汇编码，还要有对位的concept编码，所有编码索引统一从1开始，0表示特殊含义
        8. 数据集的划分，严格遵循划分原则
        9. 将所有新增的信息和映射表写入到文件中
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import defaultdict

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

# PASCAL的20个categories
# list_category_novel = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "boat", "bird", "cat",
#                        "dog", "horse", "sheep", "cow", "bottle", "chair", "couch", "potted plant", "dining table", "tv"]


def get_parameters():
    """
    原始实现：
        1. 读入dataset_coco.json
        2. 输出处理后的json文件和hdf5文件
        3. 指定单词最小频次和最大句子长度
        4. 可选指定图像根目录
    新增实现：
        1. 读入dic_coco.json，来自NBT实现
        2. 指定随机种子
        3. 指定单类最小数据量
        4. 指定novel concept个数
        5. 指定重新构造的reference json的输出位置
        6. 指定处理模式（普通模式、Few-shot模式）
    """
    print('Label Preprocess:', "parsing arguments")

    parser = argparse.ArgumentParser()
    # 文件路径
    parser.add_argument('--input_json', required=True,
                        help='input json file to process into hdf5')
    parser.add_argument('--input_dic', required=True,
                        help='input dic file to get additional info')  # 读入dic_coco.json
    parser.add_argument('--input_coco', default='data/annotations/captions_val2014.json',
                        help='input coco reference json')
    parser.add_argument('--output_json', default='label_coco.json',
                        help='output json file')
    parser.add_argument('--output_h5', default='label_coco.h5',
                        help='output h5 file')
    parser.add_argument('--output_coco', default='captions_coco.json',
                        help='output json for COCO API')
    parser.add_argument('--image_root', default='',
                        help='root location in which images are stored, to be prepended to file_path in input json')
    # 参数
    parser.add_argument('--max_length', default=16, type=int,
                        help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=5, type=int,
                        help='only words that occur more than this number of times will be put in vocab')
    parser.add_argument('--seed', default=1234, type=int,
                        help='random seed')
    parser.add_argument('--mode', default='origin', type=str, choices=['origin', 'fewshot'],
                        help='preprocess mode, make different split')
    parser.add_argument('--way', default=20, type=int,
                        help='number of novel class')
    parser.add_argument('--shot', default=10, type=int,
                        help='minimum number of shots for each blank class')
    # parser.add_argument('--superclass', default=True,
    #                     help="use superclass only")

    args = parser.parse_args()
    dict_parameters_ = vars(args)

    print('Label Preprocess:', 'parsed input parameters:')
    print(json.dumps(dict_parameters_, indent=2))

    print('Label Preprocess:', "setting seed: {:d}".format(dict_parameters_['seed']))
    random.seed(dict_parameters_['seed'])
    np.random.seed(dict_parameters_['seed'])

    return dict_parameters_


def read_files():
    """
    原始实现：
        1. 读入dataset_coco.json
    新增实现：
        1. 读入dic_coco.json，来自NBT实现
    """
    print('Label Preprocess:', "reading files")

    # 读取dataset_coco.json，获取样本信息列表images
    print('Label Preprocess:', "reading file: {}".format(dict_parameters['input_json']))
    file_dataset_coco_ = json.load(open(dict_parameters['input_json'], 'r'))

    # 读取dic_coco.json，获取wtol和wtod
    print('Label Preprocess:', "reading file: {}".format(dict_parameters['input_dic']))
    file_dic_coco_ = json.load(open(dict_parameters['input_dic'], 'r'))

    return file_dataset_coco_, file_dic_coco_


def build_dictionaries():
    """
    原始实现：
        1. 构建词汇表，过滤极低频词汇，替换为UNK（caption_filtered）
    新增实现：
        1. 从dic_coco.json中读取目标词和lemma词
        2. 包括词汇表在内的所有索引表都已dict形式输出
    """
    print('Label Preprocess:', "building maps from vocabulary, categories and concepts into indices")

    # 计算caption样本中的词频
    list_images_ = file_dataset_coco['images']
    dict_word_to_count = dict()
    for image in list_images_:
        for sentence in image['sentences']:
            for token in sentence['tokens']:
                dict_word_to_count[token] = dict_word_to_count.get(token, 0) + 1

    # 按照给定的词频阈值，选择准备替换的词汇（rare）
    list_words_vocabulary = list()
    list_words_rare = list()

    for token, num_count in dict_word_to_count.items():
        if num_count <= threshold_word_count:
            list_words_rare.append(token)
        else:
            list_words_vocabulary.append(token)

    # 词汇表统计情况
    num_words_total = sum(dict_word_to_count.values())
    num_words_rare = sum(dict_word_to_count[word] for word in list_words_rare)

    if num_words_rare > 0:
        # 存在极低频词，对这些词进行替换
        print('Label Preprocess:', 'rare words detected, inserting the special UNK token')
        list_words_vocabulary.append('UNK')

    size_words_total = len(dict_word_to_count) + 1 if num_words_rare > 0 else len(dict_word_to_count)
    size_words_vocabulary = len(list_words_vocabulary)
    size_words_rare = len(list_words_rare)

    # 基于所有的caption样本创建词汇表，从1索引，0表示空
    dict_word_to_index_ = {word: index + 1 for index, word in enumerate(list_words_vocabulary)}

    # 读取两个额外的词典，注意第一个是word maps word，第二个是word maps index，要求都从1索引
    dict_word_to_lemma_ = file_dic_coco['wtol']
    # 已经被替换的极低频词汇，直接从lemma词本中去除
    for word in dict_word_to_lemma_:
        if word not in list_words_vocabulary:
            dict_word_to_lemma_.pop(word)

    # 去掉罕见的concept（一般不会有），同时去掉莫名其妙的首尾空格
    dict_concept_to_category = {concept.strip(): category for concept, category in file_dic_coco['wtod'].items() if concept.strip() not in list_words_rare}
    # 构造concept词表，从1索引，0表示非视觉目标词
    dict_concept_to_index_ = {concept: index + 1 for index, concept in enumerate(dict_concept_to_category.keys())}

    print('Label Preprocess:', "replacing rare words by UNK token")
    for image in tqdm(list_images_):
        image['tokens_filtered'] = list()

        for sentence in image['sentences']:
            list_tokens_filtered = [token if dict_word_to_count.get(token, 0) > threshold_word_count else 'UNK' for
                                    token in sentence['tokens']]
            image['tokens_filtered'].append(list_tokens_filtered)

    print("----------------------------------------------------------------------------------------------")
    print("'Label Preprocess:', summary about vocabulary:")
    print(" - {:d} words in total".format(num_words_total))
    print(" - {:d} / {:d} ({:.2%}) different words occur less than {:d} times, will be replaced by <UNK>".format(size_words_rare, size_words_total, size_words_rare / size_words_total, threshold_word_count))
    if 'UNK' in list_words_vocabulary:
        print(" - {:d} / {:d} ({:.2%}) different words will be in vocabulary (UNK included)".format(size_words_vocabulary, size_words_total, size_words_vocabulary / size_words_total))
    else:
        print(" - {:d} / {:d} ({:.2%}) different words will be in vocabulary".format(size_words_vocabulary, size_words_total, size_words_vocabulary / size_words_total))
    print("----------------------------------------------------------------------------------------------")

    return list_images_, dict_word_to_index_, dict_word_to_lemma_, dict_concept_to_index_


def process_images():
    """
    新增实现:
        1. 记录每个图像样本包含的caption出现的concept词，按照dict_concept_to_index编码记录
        2. 如果是origin模式，则直接将karpathy的划分方案继承下来，不做改动
        3. 如果是fewshot模式，则按照我们希望的，按照BaseTrain、BaseVal、BaseTest、Support、Test的方式进行划分
        4. 划分的修改结果返回到下一步处理
    """
    # 记录每个图像中所有caption出现的concept
    print('Label Preprocess:', 'collecting concepts for images')
    for image in tqdm(list_images):
        set_index_concept = set()
        set_index_concept_full = set()

        for tokens in image['tokens_filtered']:
            for index_token, token in enumerate(tokens):
                if index_token < threshold_max_sentence_length and token in dict_concept_to_index:
                    set_index_concept.add(dict_concept_to_index[token])
                if index_token < threshold_max_sentence_length and dict_word_to_lemma[token] in dict_concept_to_index:
                    set_index_concept_full.add(dict_concept_to_index[dict_word_to_lemma[token]])

        image['list_index_concept'] = list(set_index_concept)
        image['list_index_concept_full'] = list(set_index_concept_full)

    # 预备动作：将所有的image按照类别进行划分，其中0为other类别
    print('Label Preprocess:', "split images into concepts, class OTHER for images with no concept words")
    dict_index_concept_to_index_image = defaultdict(list)

    for index_image in tqdm(range(len(list_images))):
        image = list_images[index_image]
        # other类别，不含有任何concept
        if len(image['list_index_concept']) == 0:
            dict_index_concept_to_index_image[0].append(index_image)
            continue

        for index_concept in image['list_index_concept']:
            dict_index_concept_to_index_image[index_concept].append(index_image)

    list_concept = [key
                    for key in dict_concept_to_index.keys()
                    if (' ' not in key)
                    and (len(dict_index_concept_to_index_image[dict_concept_to_index[key]]) >= dict_parameters['shot'] * 2)]  # 去掉短语和少于两倍shot量的概念

    num_concept_novel = dict_parameters['way']
    random.shuffle(list_concept)
    list_concept_novel_ = list_concept[: num_concept_novel]
    list_concept_base_ = [key
                          for key in dict_concept_to_index.keys()
                          if (' ' not in key)
                          and key not in list_concept_novel_]

    print('Label Preprocess:', "{} novel concepts:".format(num_concept_novel), list_concept_novel_)

    list_index_concept_base = [dict_concept_to_index[concept] for concept in list_concept_base_]
    list_index_concept_novel = [dict_concept_to_index[concept] for concept in list_concept_novel_]

    print('Label Preprocess:', "resplit images into concepts, class OTHER for images with no concept words")
    dict_index_concept_to_index_image.clear()
    for index_image in tqdm(range(len(list_images))):
        image = list_images[index_image]
        # other类别，不含有任何concept
        if len(image['list_index_concept']) == 0:
            dict_index_concept_to_index_image[0].append(index_image)
            continue

        # 判断一下这个image是否有novel类别
        flag_novel = False
        for index_concept in image['list_index_concept']:
            if index_concept in list_index_concept_novel:
                flag_novel = True
                break

        # 类似OD任务，像image所属的各个类别，如果这个image包含novel类别，那么不会将其归入到对应的base类别中
        for index_concept in image['list_index_concept']:
            if index_concept in list_index_concept_novel:
                dict_index_concept_to_index_image[index_concept].append(index_image)
            elif index_concept in list_index_concept_base and not flag_novel:
                dict_index_concept_to_index_image[index_concept].append(index_image)

    # 数据划分，如果为origin模式，不动，否则进行划分
    if dict_parameters['mode'] == 'fewshot':
        print('Label Preprocess:', "few-shot mode, split images into train (including val and test set for base training), support and test sets")
        # 备份一下novel的一会要删掉
        dict_index_concept_to_index_image_novel = defaultdict(list)
        for index_concept in list_index_concept_novel:
            dict_index_concept_to_index_image_novel[index_concept] = copy.deepcopy(
                dict_index_concept_to_index_image[index_concept])
            # print(index_concept, len(dict_index_concept_to_index_image[index_concept]))

        # BaseTrain、BaseVal和BaseTest，从base类别和other类别中抽取set
        set_index_image_base = set()
        set_index_image_finetune = set()
        for index_concept, list_index_image in dict_index_concept_to_index_image.items():
            if index_concept == 0 or index_concept in list_index_concept_base:
                set_index_image_base.update(list_index_image)
            else:
                set_index_image_finetune.update(list_index_image)

        # 从这个set中抽取5000个样本给BaseVal、5000个样本给BaseTest，剩下的给BaseTrain
        list_index_image_base = list(set_index_image_base)

        # coco = COCO(dict_parameters['input_coco'])
        # list_id_val = coco.getImgIds()

        # list_index_image_base_ready = [index_image for index_image in list_index_image_base if
        #                                list_images[index_image]['cocoid'] in list_id_val]
        list_index_image_base_ready = copy.deepcopy(list_index_image_base)
        random.shuffle(list_index_image_base_ready)

        list_index_image_base_val = list_index_image_base[: 5000]
        list_index_image_base_test = list_index_image_base[5000: 10000]

        list_index_image_base_train = [index_image for index_image in list_index_image_base if
                                       (index_image not in list_index_image_base_val) and (
                                               index_image not in list_index_image_base_test)]

        # 按照Karpathy的方式，给这些图片标记split，对于那些分配到novel类别的样本，同一标记为finetune（实际会使用额外的数据结构）
        print('Label Preprocess:', "split Base-Train, Base-Val and Base-Test sets")
        for index_image in tqdm(range(len(list_images))):
            image = list_images[index_image]
            if index_image in list_index_image_base_train:
                image['split'] = 'base_train'
            elif index_image in list_index_image_base_val:
                image['split'] = 'base_val'
            elif index_image in list_index_image_base_test:
                image['split'] = 'base_test'
            else:
                image['split'] = 'finetune'

        # 产生Support和Test，从novel类别中抽取，要求Test与Support不交叠
        # 先抽取Test（因为反之是允许的但是没必要）
        print("split support and test sets")
        dict_index_concept_to_list_index_image_test_ = defaultdict(list)
        set_index_image_test = set()  # 被选到test集的样本，允许重复，但是不可以在Support中出现
        for index_concept, list_index_image in dict_index_concept_to_index_image_novel.items():
            list_index_image_ready = copy.deepcopy(list_index_image)
            random.shuffle(list_index_image_ready)
            dict_index_concept_to_list_index_image_test_[index_concept] = list_index_image_ready[
                                                                          : dict_parameters['shot']]
            # print(len(list_index_image_ready[: dict_parameters['shot']]), len(dict_index_concept_to_list_index_image_test_[index_concept]))
            set_index_image_test.update(dict_index_concept_to_list_index_image_test_[index_concept])

        # 更新现有的样本，出现在test集中的样本从所有的删去，从绝对不包含任何Test集样本的剩余列表中抽取Support集
        dict_index_concept_to_list_index_image_support_ = defaultdict(list)
        set_index_image_support = set()
        for index_concept, list_index_image in dict_index_concept_to_index_image_novel.items():
            list_index_image_new = [index_image for index_image in list_index_image if
                                    index_image not in set_index_image_test]
            random.shuffle(list_index_image_new)
            dict_index_concept_to_list_index_image_support_[index_concept] = list_index_image_new[
                                                                             : dict_parameters['shot']]
            # print(len(list_index_image_new[: dict_parameters['shot']]), len(dict_index_concept_to_list_index_image_support_[index_concept]))
            set_index_image_support.update(dict_index_concept_to_list_index_image_support_[index_concept])

        # print(len(set_index_image_support), len(set_index_image_test))

        dict_index_word_to_count_base = dict()
        # 额外动作：计算一下两个大的split的词汇表差异
        for index_image in list_index_image_base:
            for tokens in list_images[index_image]['tokens_filtered']:
                for token in tokens:
                    dict_index_word_to_count_base[token] = dict_index_word_to_count_base.get(token, 0) + 1

        list_index_image_finetune = list(set.union(set_index_image_support, set_index_image_test))
        dict_index_word_to_count_finetune = dict()

        for index_image in list_index_image_finetune:
            for tokens in list_images[index_image]['tokens_filtered']:
                for token in tokens:
                    dict_index_word_to_count_finetune[token] = dict_index_word_to_count_finetune.get(token, 0) + 1

        print('Label Preprocess:', "BASE VOCABULARY SIZE: {}".format(len(dict_index_word_to_count_base)))

        print('Label Preprocess:', "NOVEL VOCABULARY SIZE: {}".format(len(dict_index_word_to_count_finetune)))

        return \
            list_images, \
            dict_index_concept_to_list_index_image_support_, \
            dict_index_concept_to_list_index_image_test_, \
            list_concept_base_, \
            list_concept_novel_, \
            len(set_index_image_support), \
            len(set_index_image_test)

    else:
        print('Label Preprocess:', "origin mode, reuse Karpathy's split")
        return list_images, None, None, None, None, None, None


def encode_captions():
    """
    原始实现：
        1. 编码captions到index vectors，构成一个矩阵，用0补齐
        2. 记录每一个image对应的编码向量的起始和结束索引位（处理的时候，每一张图像的所有caption都是连续处理的，所以不会出问题）
        3. 记录每一个caption的长度（我个人认为这个没啥用，生成mask相对快一些？）
    """
    print("encoding captions into vectors")

    num_images = len(list_images_new)
    num_captions = sum(len(image['tokens_filtered']) for image in list_images_new)

    # 这里有一个历史包袱：索引是Lua的风格，从1开始（因为最早的实现使用Lua处理）
    list_label_samples = list()
    label_start_index_ = np.zeros(num_images, dtype='uint32')
    label_end_index_ = np.zeros(num_images, dtype='uint32')
    label_length_ = np.zeros(num_captions, dtype='uint32')

    # caption编码
    counter_caption = 0
    counter_caption_lua = 1
    for index_image in tqdm(range(len(list_images_new))):
        image = list_images_new[index_image]
        num_captions_image = len(image['tokens_filtered'])
        assert num_captions_image > 0, 'error: some image has no captions !'

        label_sample = np.zeros((num_captions_image, threshold_max_sentence_length), dtype='uint32')

        for index_caption, caption in enumerate(image['tokens_filtered']):

            label_length_[counter_caption] = min(threshold_max_sentence_length, len(caption))
            for index_token, token in enumerate(caption):
                if index_token < threshold_max_sentence_length:
                    label_sample[index_caption, index_token] = dict_word_to_index[token]

            counter_caption += 1

        list_label_samples.append(label_sample)
        label_start_index_[index_image] = counter_caption_lua
        label_end_index_[index_image] = counter_caption_lua + num_captions_image - 1
        counter_caption_lua += num_captions_image

    label_ = np.concatenate(list_label_samples, axis=0)

    assert label_.shape[0] == num_captions, 'lengths don\'t match? that\'s weird'
    assert np.all(label_length_ > 0), 'error: some caption had no words?'

    print('Label Preprocess:', 'encoded captions to array of size {}'.format(label_.shape))

    return (
        label_,
        label_start_index_,
        label_end_index_,
        label_length_,
    )


def write_files():
    """
    原始实现：
        1. 将词汇表写入json文件
        2. 将图像的基本信息（划分、id、图像路径）写入json
        3. 将编码后的caption写入hdf5文件
        4. 将每个样本对应的区段索引以及各编码向量的实际长度写入hdf5文件
    新增实现：
        1. 将concept表、lemma表、caption与image的映射表写入json文件
        2. 将划分好的concept类别等信息写入到json文件
        3. 将数据子集的划分信息写入到json文件
    """
    print("writing infos into files")

    file_hdf5 = h5py.File(dict_parameters['output_h5'], "w")
    file_hdf5.create_dataset("label", dtype='uint32', data=label)
    file_hdf5.create_dataset("label_start_index", dtype='uint32', data=label_start_index)
    file_hdf5.create_dataset("label_end_index", dtype='uint32', data=label_end_index)
    file_hdf5.create_dataset("label_length", dtype='uint32', data=label_length)
    file_hdf5.close()

    file_json = dict()
    file_json['dict_index_to_word'] = {index: word for word, index in dict_word_to_index.items()}
    file_json['dict_index_to_concept'] = {index: word for word, index in dict_concept_to_index.items()}
    file_json['dict_word_to_lemma'] = dict_word_to_lemma
    file_json['list_concept_base'] = list_concept_base
    file_json['list_concept_novel'] = list_concept_novel
    file_json['dict_index_concept_to_list_index_image_support'] = dict_index_concept_to_list_index_image_support
    file_json['dict_index_concept_to_list_index_image_test'] = dict_index_concept_to_list_index_image_test

    file_json['list_images'] = list()

    dict_size_split = dict()
    for index_image, image in enumerate(list_images_new):
        info_image = dict()
        info_image['split'] = image['split']
        dict_size_split[image['split']] = dict_size_split.get(image['split'], 0) + 1

        if 'filename' in image:
            info_image['file_path'] = os.path.join(image.get('filepath', ''), image['filename'])

        info_image['image_id'] = image['cocoid'] if 'cocoid' in image else image['imgid']

        # 如果需要使用包围盒信息，这部分必须要读取，否则会确实必要信息
        if dict_parameters['image_root'] != '':
            with Image.open(
                    os.path.join(dict_parameters['image_root'], image['filepath'], image['filename'])) as image_ready:
                info_image['width'], info_image['height'] = image_ready.size

        info_image['list_index_concept'] = image['list_index_concept']

        file_json['list_images'].append(info_image)

    print("----------------------------------------------------------------------------------------------")
    print('Label Preprocess:', 'summary about split:')
    for key, value in dict_size_split.items():
        if key == 'finetune':
            print(" - split {} : {} images, {} images for support and {} images for test".format(key, value,
                                                                                                 size_index_image_support,
                                                                                                 size_index_image_test))
        else:
            print(" - split {} : {} images".format(key, value))

    print("----------------------------------------------------------------------------------------------")

    json.dump(file_json, open(dict_parameters['output_json'], 'w'))

    if dict_parameters['mode'] == 'fewshot':
        print('Label Preprocess:', "few-shot mode, creating new reference json file for COCO API")
        # 搞定COCO API参考文件，将base-val和base-test对应地样本按照COCO API格式生成json文件用于读取
        file_coco = dict()

        # info、type还有license，不知道会不会使用，能复制的直接复制过来
        file_coco['info'] = {
            'description': 'References of Base_val split and Base_test split for COCO API',
            'url': 'https://github.com/caryleo',
            'version': '1.0',
            'year': 2020,
            'contributor': 'Hao Liu',
            'date_created': time.strftime("%Y/%m/%d")
        }

        file_coco['type'] = 'captions'

        file_coco['licenses'] = [
            {'url': 'http://cocodataset.org', 'id': 1, 'name': 'All images from MS COCO dataset'},
        ]

        # images和annotations 置空，准备写入
        file_coco['images'] = list()
        file_coco['annotations'] = list()

        # 将base-val和base-test中样本对用的image信息包含过来
        list_index_images_coco = [index_image for index_image, image in enumerate(list_images_new) if
                                  image['split'] == 'base_val' or image['split'] == 'base_test']

        for index_image in list_index_images_coco:
            image = list_images_new[index_image]
            file_coco['images'].append(
                {
                    'id': image['cocoid'] if 'cocoid' in image else image['imgid'],
                    'url': 'http://cocodataset.org',
                    'file_name': image['filename']
                }
            )

            for instance_caption in image['sentences']:
                file_coco['annotations'].append(
                    {
                        'image_id': image['cocoid'] if 'cocoid' in image else image['imgid'],
                        'id': instance_caption['sentid'],
                        'caption': instance_caption['raw']
                    }
                )

        json.dump(file_coco, open(dict_parameters['output_coco'], 'w'))

    print("All Done!")


dict_parameters = get_parameters()

threshold_word_count = dict_parameters['word_count_threshold']
threshold_max_sentence_length = dict_parameters['max_length']

file_dataset_coco, file_dic_coco = read_files()

(
    list_images,
    dict_word_to_index,
    dict_word_to_lemma,
    dict_concept_to_index,
) = build_dictionaries()

(
    list_images_new,
    dict_index_concept_to_list_index_image_support,
    dict_index_concept_to_list_index_image_test,
    list_concept_base,
    list_concept_novel,
    size_index_image_support,
    size_index_image_test
) = process_images()

(
    label,
    label_start_index,
    label_end_index,
    label_length,
) = encode_captions()

write_files()
