"""
原始实现：
    1. 读取文件，并根据文件中指定的图像信息读取图像实例
    2. 读取图像，并使用与训练ResNet网络模型处理图像获得特征
    3. 根据需要，保存特定的图像特征到文件中，现有实现按照图像为单位保存
新增实现：
    1. 仅修改了部分命名和局部处理流程
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import argparse
from random import shuffle, seed
import string
# non-standard dependencies:
import h5py
from six.moves import cPickle
import numpy as np
import torch
import torchvision.models as models
import skimage.io

from torchvision import transforms as trn

preprocess = trn.Compose([
    # trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from captioning.utils.resnet_utils import myResnet
import captioning.utils.resnet as resnet


def main(params):
    net = getattr(resnet, params['model'])()
    net.load_state_dict(torch.load(os.path.join(params['model_root'], params['model'] + '.pth')))
    my_resnet = myResnet(net)
    my_resnet.cuda()
    my_resnet.eval()

    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']
    N = len(imgs)

    seed(123)  # make reproducible

    dir_fc = params['output_dir'] + '_fc'
    dir_att = params['output_dir'] + '_att'
    if not os.path.isdir(dir_fc):
        os.mkdir(dir_fc)
    if not os.path.isdir(dir_att):
        os.mkdir(dir_att)

    for i, img in enumerate(imgs):
        # load the image
        I = skimage.io.imread(os.path.join(params['images_root'], img['filepath'], img['filename']))
        # handle grayscale input images
        if len(I.shape) == 2:
            I = I[:, :, np.newaxis]
            I = np.concatenate((I, I, I), axis=2)

        I = I.astype('float32') / 255.0
        I = torch.from_numpy(I.transpose([2, 0, 1])).cuda()
        I = preprocess(I)
        with torch.no_grad():
            tmp_fc, tmp_att = my_resnet(I, params['att_size'])
        # write to pkl
        np.save(os.path.join(dir_fc, str(img['cocoid'])), tmp_fc.data.cpu().float().numpy())
        np.savez_compressed(os.path.join(dir_att, str(img['cocoid'])), feat=tmp_att.data.cpu().float().numpy())

        if i % 1000 == 0:
            print('Grid Feature Preprocess:', 'processing {}/{} images ({:.2f%} done)'.format(i, N, i * 100.0 / N))

    print('wrote ', params['output_dir'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', required=True, help='input json file to process into hdf5')
    parser.add_argument('--output_dir', default='data', help='output h5 file')

    # options
    parser.add_argument('--images_root', default='',
                        help='root location in which images are stored, to be prepended to file_path in input json')
    parser.add_argument('--att_size', default=14, type=int, help='14x14 or 7x7')
    parser.add_argument('--model', default='resnet101', type=str, help='resnet101, resnet152')
    parser.add_argument('--model_root', default='./data/imagenet_weights', type=str, help='model root')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)
