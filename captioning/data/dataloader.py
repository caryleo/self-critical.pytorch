"""
原始实现：
    1. 图像特征的加载实现，能够适应不同类型的特征文件的读取和加载
    2. 图像特征的加载实现，能够实现in-memory读取模式
    3. 数据集实例实现，包含对信息和映射表的读取
    4. 数据集实例实现，包含对图像划分的整理（传统划分，训练+验证+测试）
    5. 数据集实例实现，包含对图像及其caption样本的读取和batch处理
    6. 数据加载器实现，根据需要初始化各个数据自己的数据集实例
    7. 数据加载器实现，封装batch处理和状态信息加载
    8. 采样器实例实现，处理训练数据集的跨轮次无界batch构造
    9. 采样器实例实现，状态信息加载
新增实现：
    1. 数据集实例，新增的信息表和映射表的读取
    2. 数据集实例，修改部分命名以使既有数据集划分处理对应现有经典子集
    3. 数据集实例，增加对支持集和测试集的封装处理
    4. 数据集实例，增加对支持集和测试集数据处理和batch处理时的对concept的处理
    5. 数据加载器，调整了各个split的实例化实现逻辑（原有实现有异步实例化bug）
    6. 采样器，调整了对于新增两个数据子集读取过程中索引的开箱处理
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import lmdb
import os
import numpy as np
import numpy.random as npr
import random

import torch
import torch.utils.data as data

import multiprocessing
import six


class HybridLoader:
    """
    原始实现：
        1. 封装不同文件形式的图像特征读取实现
        2. 文件路径指定，常规按照图像样本读取
        3. lmdb格式，使用lmdb工具进行特征读取
        4. pth格式，使用torch读取工具进行特征读取
        5. h5格式，使用hdf5工具进行特征读取
        6. in-memory数据加载方式实现（对于后三种形式）
    """
    def __init__(self, db_path, ext, in_memory=False):
        self.db_path = db_path
        self.ext = ext
        if self.ext == '.npy':
            self.loader = lambda x: np.load(six.BytesIO(x))
        else:
            def load_npz(x):
                x = np.load(six.BytesIO(x))
                return x['feat'] if 'feat' in x else x['z']  # 作者注：原始数据文件存在错误

            self.loader = load_npz
        if db_path.endswith('.lmdb'):
            self.db_type = 'lmdb'
            self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                                 readonly=True, lock=False,
                                 readahead=False, meminit=False)
        elif db_path.endswith('.pth'):  # 作者注：pth文件实际上就是类似dict的键值对机制
            self.db_type = 'pth'
            self.feat_file = torch.load(db_path)
            self.loader = lambda x: x
            print('HybridLoader: ext is ignored')
        elif db_path.endswith('h5'):
            self.db_type = 'h5'
            self.loader = lambda x: np.array(x).astype('float32')
        else:
            self.db_type = 'dir'

        self.in_memory = in_memory
        if self.in_memory:
            self.features = {}

    def get(self, key):
        if self.in_memory and key in self.features:
            f_input = self.features[key]
        elif self.db_type == 'lmdb':
            env = self.env
            with env.begin(write=False) as txn:
                byteflow = txn.get(key.encode())
            f_input = byteflow
        elif self.db_type == 'pth':
            f_input = self.feat_file[key]
        elif self.db_type == 'h5':
            f_input = h5py.File(self.db_path, 'r')[key]
        else:
            f_input = open(os.path.join(self.db_path, key + self.ext), 'rb').read()

        if self.in_memory and key not in self.features:
            self.features[key] = f_input

        # load image
        feat = self.loader(f_input)

        return feat


class Dataset(data.Dataset):
    """
    原始实现：
        1. 从指定的文件中读取样本和信息表信息
        2. 对图像既定样本划分的整理（传统：训练+验证+测试）
        3. 对单个样本的特征读取
        4. 对batch数据的处理
    新增实现：
        1. 对新增的信息表和映射表的读取
        2. 对新增支持集和测试机划分的读取
        3. 对于新增数据子集关于concept信息的处理
        4. 对于新增数据子集关于concept信息的batch处理
        5. 修改train_only策略：当置1时，将支持集样本汇入到训练集中
    """

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.seq_per_img = opt.seq_per_img

        self.use_fc = getattr(opt, 'use_fc', True)
        self.use_att = getattr(opt, 'use_att', True)
        self.use_box = getattr(opt, 'use_box', 0)
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)

        # 读取必要的文件及其中的信息
        print('DataLoader:', 'loading json file: {}'.format(opt.input_json))
        self.info = json.load(open(self.opt.input_json))
        if 'dict_index_to_word' in self.info:
            self.ix_to_word = self.info['dict_index_to_word']
            self.vocab_size = len(self.ix_to_word)
            print('DataLoader:', 'vocab size is {}'.format(self.vocab_size))

        if 'dict_index_to_concept' in self.info:
            self.ix_to_concept = self.info['dict_index_to_concept']
            self.concept_size = len(self.ix_to_concept)
            print('DataLoader:', 'concept size is {}'.format(self.concept_size))

        if 'list_concept_base' in self.info:
            self.concept_base = self.info['list_concept_base']
            self.concept_base_size = len(self.concept_base)
            print('DataLoader:', 'base concept size is {}'.format(self.concept_base_size))

        if 'list_concept_novel' in self.info:
            self.concept_novel = self.info['list_concept_novel']
            self.concept_novel_size = len(self.concept_novel)
            print('DataLoader:', 'novel concept size is {}'.format(self.concept_novel_size))

        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_box_dir, opt.input_label_h5)

        # 作者注：有一些情况下我们没有GT的caption输入
        if self.opt.input_label_h5 != 'none':
            self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')
            # load in the sequence data
            seq_size = self.h5_label_file['label'].shape
            self.label = self.h5_label_file['label'][:]
            self.seq_length = seq_size[1]
            print('Dataloader:', 'max sequence length in data is', self.seq_length)
            # load the pointers in full to RAM (should be small enough)
            self.label_start_ix = self.h5_label_file['label_start_index'][:]
            self.label_end_ix = self.h5_label_file['label_end_index'][:]
        else:
            self.seq_length = 1

        self.data_in_memory = getattr(opt, 'data_in_memory', False)
        self.fc_loader = HybridLoader(self.opt.input_fc_dir, '.npy', in_memory=self.data_in_memory)
        self.att_loader = HybridLoader(self.opt.input_att_dir, '.npz', in_memory=self.data_in_memory)
        self.box_loader = HybridLoader(self.opt.input_box_dir, '.npy', in_memory=self.data_in_memory)

        self.num_images = len(self.info['list_images'])
        print('Dataloader:', 'read {} image features'.format(self.num_images))

        # 新增数据子集：支持集和测试机
        self.dict_index_concept_to_list_index_image_support = self.info['dict_index_concept_to_list_index_image_support']
        self.dict_index_concept_to_list_index_image_test = self.info['dict_index_concept_to_list_index_image_test']

        # 划分处理
        if opt.train_only == 0:
            self.split_ix = {'base_train': [], 'base_val': [], 'base_test': [], 'test': []}
        else:
            self.split_ix = {'base_train': [], 'base_val': [], 'base_test': [], 'support': [], 'test': []}

        for ix in range(len(self.info['list_images'])):
            img = self.info['list_images'][ix]
            if not 'split' in img:
                self.split_ix['base_train'].append(ix)
                self.split_ix['base_val'].append(ix)
                self.split_ix['base_test'].append(ix)
            elif img['split'] == 'base_train':
                self.split_ix['base_train'].append(ix)
            elif img['split'] == 'base_val':
                self.split_ix['base_val'].append(ix)
            elif img['split'] == 'base_test':
                self.split_ix['base_test'].append(ix)

        # 修改了train_only的策略，置0时，将support集中的样本汇入到训练集中
        if opt.train_only == 0:
            for ix_concept, ix_images in self.dict_index_concept_to_list_index_image_support.items():
                for ix in ix_images:
                    if ix not in self.split_ix['base_train']:
                        self.split_ix['base_train'].append(ix)
        else:
            for ix_concept, ix_images in self.dict_index_concept_to_list_index_image_support.items():
                for ix in ix_images:
                    self.split_ix['support'].append({'ix_concept': int(ix_concept), 'ix_image': ix})

        for ix_concept, ix_images in self.dict_index_concept_to_list_index_image_test.items():
            for ix in ix_images:
                self.split_ix['test'].append({'ix_concept': int(ix_concept), 'ix_image': ix})

        for key, list_item in self.split_ix.items():
            print('DataLoader:', 'assigned {:d} images to split {}'.format(len(self.split_ix[key]), key))

        # print('assigned %d images to split base_train' %len(self.split_ix['base_train']))
        # print('assigned %d images to split base_val' %len(self.split_ix['base_val']))
        # print('assigned %d images to split base_test' %len(self.split_ix['base_test']))
        # print('assigned %d images to split support' %len(self.split_ix['support']))
        # print('assigned %d images to split test' %len(self.split_ix['test']))

    def get_captions(self, ix, seq_per_img):
        # 作者注：读入位置信息，注意这里的位置信息是从1开始的，实际使用需要减一
        ix1 = self.label_start_ix[ix] - 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype='int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1, ix2)
                seq[q, :] = self.label[ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.label[ixl: ixl + seq_per_img, :self.seq_length]

        return seq

    def collate_func(self, batch, split):
        seq_per_img = self.seq_per_img

        concept_batch = []

        fc_batch = []
        att_batch = []
        label_batch = []

        wrapped = False

        infos = []
        gts = []

        for sample in batch:
            tmp_ix_concept, tmp_fc, tmp_att, tmp_seq, ix, it_pos_now, tmp_wrapped = sample

            if tmp_wrapped:
                wrapped = True

            concept_batch.append(tmp_ix_concept)

            fc_batch.append(tmp_fc)
            att_batch.append(tmp_att)

            # 作者注：读取GT caption，考虑没有GT的情形
            tmp_label = np.zeros([seq_per_img, self.seq_length + 2], dtype='int')
            if hasattr(self, 'h5_label_file'):
                tmp_label[:, 1: self.seq_length + 1] = tmp_seq
            label_batch.append(tmp_label)

            # 作者注：读取GT caption，用于SCST中计算reward，同样需要考虑可能没有GT的情形（跳过）
            if hasattr(self, 'h5_label_file'):
                gts.append(self.label[self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
            else:
                gts.append([])

            # 图像信息，如果concept非空，说明支持集或测试集，导出concept信息
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['list_images'][ix]['image_id']
            info_dict['file_path'] = self.info['list_images'][ix].get('file_path', '')
            if tmp_ix_concept is not None:
                info_dict['list_concepts'] = self.info['list_images'][ix]['list_index_concept']

            infos.append(info_dict)

        # #sort by att_feat length
        # fc_batch, att_batch, label_batch, gts, infos = \
        #     zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, infos), key=lambda x: len(x[1]), reverse=True))
        concept_batch, fc_batch, att_batch, label_batch, gts, infos = \
            zip(*sorted(zip(concept_batch, fc_batch, att_batch, label_batch, gts, infos), key=lambda x: 0,reverse=True))

        data = {}

        data['concepts'] = concept_batch

        data['fc_feats'] = np.stack(fc_batch)
        max_att_len = max([_.shape[0] for _ in att_batch])
        data['att_feats'] = np.zeros([len(att_batch), max_att_len, att_batch[0].shape[1]], dtype='float32')
        for i in range(len(att_batch)):
            data['att_feats'][i, :att_batch[i].shape[0]] = att_batch[i]
        data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
        for i in range(len(att_batch)):
            data['att_masks'][i, :att_batch[i].shape[0]] = 1
        # 作者注：如果所有的样本的att特征个数相同（updown情形），则将mask直接置空
        if data['att_masks'].sum() == data['att_masks'].size:
            data['att_masks'] = None

        data['labels'] = np.vstack(label_batch)
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, data['labels'])))
        mask_batch = np.zeros([data['labels'].shape[0], self.seq_length + 2], dtype='float32')
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch
        data['labels'] = data['labels'].reshape(len(batch), seq_per_img, -1)
        data['masks'] = data['masks'].reshape(len(batch), seq_per_img, -1)

        data['gts'] = gts
        data['bounds'] = {'it_pos_now': it_pos_now,
                          'it_max': len(self.split_ix[split]),
                          'wrapped': wrapped}
        data['infos'] = infos

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in data.items()}

        return data

    def __getitem__(self, index):
        ix_concept, ix, it_pos_now, wrapped = index  # self.split_ix[index]
        if self.use_att:
            att_feat = self.att_loader.get(str(self.info['list_images'][ix]['image_id']))
            # 作者注：整理成二维向量
            att_feat = att_feat.reshape(-1, att_feat.shape[-1])
            if self.norm_att_feat:
                att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)
            if self.use_box:
                box_feat = self.box_loader.get(str(self.info['list_images'][ix]['image_id']))
                # 作者注：处理成标准化相对位置和相对尺寸信息
                x1, y1, x2, y2 = np.hsplit(box_feat, 4)
                h, w = self.info['list_images'][ix]['height'], self.info['list_images'][ix]['width']
                box_feat = np.hstack(
                    (x1 / w, y1 / h, x2 / w, y2 / h, (x2 - x1) * (y2 - y1) / (w * h)))  # question? x2-x1+1??
                if self.norm_box_feat:
                    box_feat = box_feat / np.linalg.norm(box_feat, 2, 1, keepdims=True)
                att_feat = np.hstack([att_feat, box_feat])
                # 作者注：按照尺寸进行排序
                att_feat = np.stack(sorted(att_feat, key=lambda x: x[-1], reverse=True))
        else:
            att_feat = np.zeros((0, 0), dtype='float32')
        if self.use_fc:
            try:
                fc_feat = self.fc_loader.get(str(self.info['list_images'][ix]['image_id']))
            except:
                # 作者注：在updown情形，全连接特征使用att特征的平均值结果
                fc_feat = att_feat.mean(0)
        else:
            fc_feat = np.zeros((0), dtype='float32')
        if hasattr(self, 'h5_label_file'):
            seq = self.get_captions(ix, self.seq_per_img)
        else:
            seq = None

        return (ix_concept,
                fc_feat, att_feat, seq,
                ix, it_pos_now, wrapped)

    def __len__(self):
        return len(self.info['list_images'])


class DataLoader:
    """
    原始实现：
        1. 各个数据子集对应地数据集实例化
        2. 一些信息读取的封装
        3. 信息加载封装
    新增实现：
        1. 调整了实例化过程（原始逻辑存在异步实例化bug）
    """
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.batch_size_finetune = self.opt.batch_size_finetune
        self.dataset = Dataset(opt)

        self.loaders, self.iters = {}, {}

        # base_train
        sampler_base_train = MySampler(self.dataset.split_ix['base_train'], shuffle=True, wrap=True)
        self.loaders['base_train'] = data.DataLoader(dataset=self.dataset,
                                                     batch_size=self.batch_size,
                                                     sampler=sampler_base_train,
                                                     pin_memory=True,
                                                     num_workers=4,
                                                     collate_fn=lambda x: self.dataset.collate_func(x, 'base_train'),
                                                     drop_last=False)
        self.iters['base_train'] = iter(self.loaders['base_train'])

        # base_val
        sampler_base_val = MySampler(self.dataset.split_ix['base_val'], shuffle=False, wrap=False)
        self.loaders['base_val'] = data.DataLoader(dataset=self.dataset,
                                                   batch_size=self.batch_size,
                                                   sampler=sampler_base_val,
                                                   pin_memory=True,
                                                   num_workers=4,
                                                   collate_fn=lambda x: self.dataset.collate_func(x, 'base_val'),
                                                   drop_last=False)
        self.iters['base_val'] = iter(self.loaders['base_val'])

        # base_test
        sampler_base_test = MySampler(self.dataset.split_ix['base_test'], shuffle=False, wrap=False)
        self.loaders['base_test'] = data.DataLoader(dataset=self.dataset,
                                                    batch_size=self.batch_size,
                                                    sampler=sampler_base_test,
                                                    pin_memory=True,
                                                    num_workers=4,
                                                    collate_fn=lambda x: self.dataset.collate_func(x, 'base_test'),
                                                    drop_last=False)
        self.iters['base_test'] = iter(self.loaders['base_test'])

        # support，train_only = 1时，可能没有支持集
        if 'support' in self.dataset.split_ix:
            sampler_support = MySampler(self.dataset.split_ix['support'], shuffle=True, wrap=True, dual=True)
            self.loaders['support'] = data.DataLoader(dataset=self.dataset,
                                                      batch_size=self.batch_size_finetune,
                                                      sampler=sampler_support,
                                                      pin_memory=True,
                                                      num_workers=1,
                                                      collate_fn=lambda x: self.dataset.collate_func(x, 'support'),
                                                      drop_last=False)
            self.iters['support'] = iter(self.loaders['support'])

        # test
        sampler_test = MySampler(self.dataset.split_ix['test'], shuffle=False, wrap=False, dual=True)
        self.loaders['test'] = data.DataLoader(dataset=self.dataset,
                                               batch_size=self.batch_size_finetune,
                                               sampler=sampler_test,
                                               pin_memory=True,
                                               num_workers=1,
                                               collate_fn=lambda x: self.dataset.collate_func(x, 'test'),
                                               drop_last=False)
        self.iters['test'] = iter(self.loaders['test'])

    def get_batch(self, split):
        try:
            data = next(self.iters[split])
        except StopIteration:
            self.iters[split] = iter(self.loaders[split])
            data = next(self.iters[split])
        return data

    def reset_iterator(self, split):
        self.loaders[split].sampler._reset_iter()
        self.iters[split] = iter(self.loaders[split])

    def get_vocab_size(self):
        return self.dataset.get_vocab_size()

    @property
    def vocab_size(self):
        return self.get_vocab_size()

    def get_vocab(self):
        return self.dataset.get_vocab()

    def get_seq_length(self):
        return self.dataset.get_seq_length()

    @property
    def seq_length(self):
        return self.get_seq_length()

    @property
    def concept_novel(self):
        return self.dataset.concept_novel

    def state_dict(self):
        def get_prefetch_num(split):
            if self.loaders[split].num_workers > 0:
                return (self.iters[split]._send_idx - self.iters[split]._rcvd_idx) * self.batch_size
            else:
                return 0

        return {split: loader.sampler.state_dict(get_prefetch_num(split)) \
                for split, loader in self.loaders.items()}

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        for split in self.loaders.keys():
            self.loaders[split].sampler.load_state_dict(state_dict[split])


class MySampler(data.sampler.Sampler):
    """
    原始实现：
        1. 训练集的跨轮次无界取batch实现
        2. 常规采样器实现
    新增实现：
        1. 对新增数据子集的索引开箱处理
    """
    def __init__(self, index_list, shuffle, wrap, dual=False):
        self.index_list = index_list
        self.shuffle = shuffle
        self.wrap = wrap
        self.dual = dual
        self._reset_iter()

    def __iter__(self):
        return self

    def __next__(self):
        wrapped = False
        if self.iter_counter == len(self._index_list):
            self._reset_iter()
            if self.wrap:
                wrapped = True
            else:
                raise StopIteration()
        if len(self._index_list) == 0:  # overflow when 0 samples
            return None

        if self.dual:
            elem = (self._index_list[self.iter_counter]['ix_concept'], self._index_list[self.iter_counter]['ix_image'], self.iter_counter + 1, wrapped)
        else:
            elem = (None, self._index_list[self.iter_counter], self.iter_counter + 1, wrapped)

        self.iter_counter += 1
        return elem

    def next(self):
        return self.__next__()

    def _reset_iter(self):
        if self.shuffle:
            rand_perm = npr.permutation(len(self.index_list))
            self._index_list = [self.index_list[_] for _ in rand_perm]
        else:
            self._index_list = self.index_list

        self.iter_counter = 0

    def __len__(self):
        return len(self.index_list)

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        self._index_list = state_dict['index_list']
        self.iter_counter = state_dict['iter_counter']

    def state_dict(self, prefetched_num=None):
        prefetched_num = prefetched_num or 0
        return {
            'index_list': self._index_list,
            'iter_counter': self.iter_counter - prefetched_num
        }
