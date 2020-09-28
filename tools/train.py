"""
原始实现：
    1. 初始化训练迭代信息，并在必要的时候从既有保存文件中加载
    2. 初始化日志，实现文件日志和tensorboard日志，并在必要的时候从既有保存文件中加载
    3. 初始化模型，并在必要的时候从既有保存文件中加载
    4. 初始化并配置数据加载器，并在必要的时候加载迭代状态
    5. 完成训练，在必要的时候保存当前模型迭代信息，以及在必要的时候完成模型的选择评估
    6. 在合适的时候对现有模型进行保存
新增实现：
    1. 新增了训练阶段信息，1为经典训练阶段，2为Finetune阶段
    2. 将既有的训练流程调整为经典训练阶段，新增Finetune阶段，该阶段内模型不进行评估选择
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import time
import os
from six.moves import cPickle
import traceback
from collections import defaultdict

import captioning.utils.opts as opts
import captioning.models as models
from captioning.data.dataloader import *
import skimage.io
import captioning.utils.eval_utils as eval_utils
import captioning.utils.misc as utils
from captioning.utils.rewards import init_scorer, get_self_critical_reward
from captioning.modules.loss_wrapper import LossWrapper


def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)


def train(opt):
    ################################
    # 创建dataloader
    ################################
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length

    ##########################
    # 初始化训练信息
    ##########################
    infos = {
        'iter': 0,
        'epoch': 0,
        'loader_state_dict': None,
        'vocab': loader.get_vocab(),
        'stage': 1,
        'stage_saved': 1  # 用于中断处理，记录了中断时的状态，用于判定是否重新加载最佳模型
    }

    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl')):
        with open(os.path.join(opt.start_from, 'infos_' + opt.id + '.pkl'), 'rb') as f:
            infos = utils.pickle_load(f)
            saved_model_opt = infos['opt']
            need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert getattr(saved_model_opt, checkme) == getattr(opt, checkme), "Command line argument and saved model disagree on '%s' " % checkme
    infos['opt'] = opt

    #########################
    # 创建logger
    #########################
    # 文件logger
    histories = defaultdict(dict)
    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl')):
        with open(os.path.join(opt.start_from, 'histories_' + opt.id + '.pkl'), 'rb') as f:
            histories.update(utils.pickle_load(f))

    # tensorboard logger
    tb_summary_writer = SummaryWriter(opt.checkpoint_path)

    ##########################
    # 创建模型
    ##########################
    opt.vocab = loader.get_vocab()
    model = models.setup(opt).cuda()
    del opt.vocab

    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, 'model.pth')):
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))

    # 作者注：面向模型的loss封装，便于将loss计算独立，便于多卡时减小No.0 GPU的负载
    lw_model = LossWrapper(model, opt)
    # 多GPU封装
    dp_model = torch.nn.DataParallel(model)
    dp_model.vocab = getattr(model, 'vocab', None)
    dp_lw_model = torch.nn.DataParallel(lw_model)

    model.set_stage(infos['stage'])

    ##########################
    #  创建优化器
    ##########################
    if opt.noamopt:
        assert opt.caption_model in ['transformer', 'bert', 'm2transformer'], 'noamopt can only work with transformer'
        optimizer = utils.get_std_opt(model, optim_func=opt.optim, factor=opt.noamopt_factor, warmup=opt.noamopt_warmup)
    elif opt.reduce_on_plateau:
        optimizer = utils.build_optimizer(model.parameters(), opt)
        optimizer = utils.ReduceLROnPlateau(optimizer,
                                            factor=opt.reduce_on_plateau_factor,
                                            patience=opt.reduce_on_plateau_patience)
    else:
        optimizer = utils.build_optimizer(model.parameters(), opt)

    if opt.start_from is not None and os.path.isfile(os.path.join(opt.start_from, "optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    #########################
    # 训练
    #########################

    # 准备阶段
    iteration = infos['iter']
    epoch = infos['epoch']
    loader.load_state_dict(infos['loader_state_dict'])
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)
    if opt.noamopt:
        optimizer._step = iteration

    # 作者注：轮次完成标志量，用于新轮次可能的训练参数调整
    epoch_done = True
    eval_done = False

    dp_lw_model.train()

    # 开始训练啦！经典训练
    if infos['stage'] == 1:
        try:
            while True:
                # 达到最大epoch限制，跳出经典训练
                if epoch >= opt.max_epochs_base != -1:
                    if eval_done:
                        break
                    else:
                        # 末尾再评估一次
                        eval_kwargs = {'split': 'base_val', 'dataset': opt.input_json}
                        eval_kwargs.update(vars(opt))
                        val_loss, predictions, lang_stats = eval_utils.eval_split(dp_model, lw_model.crit, loader, eval_kwargs)

                        if opt.reduce_on_plateau:
                            if 'CIDEr' in lang_stats:
                                optimizer.scheduler_step(-lang_stats['CIDEr'])
                            else:
                                optimizer.scheduler_step(val_loss)

                        # 将评估结果写入日志
                        tb_summary_writer.add_scalar('validation loss', val_loss, iteration)
                        if lang_stats is not None:
                            for k, v in lang_stats.items():
                                tb_summary_writer.add_scalar(k, v, iteration)

                        histories['val_result_history'][iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

                        # 根据CIDEr指标选择最佳模型
                        if opt.language_eval == 1:
                            current_score = lang_stats['CIDEr']
                        else:
                            current_score = - val_loss

                        best_flag = False

                        if best_val_score is None or current_score > best_val_score:
                            best_val_score = current_score
                            best_flag = True

                        infos['best_val_score'] = best_val_score

                        utils.save_checkpoint(opt, model, infos, optimizer, histories)

                        if opt.save_history_ckpt:
                            utils.save_checkpoint(opt, model, infos, optimizer, append=str(epoch) if opt.save_every_epoch else str(iteration))

                        if best_flag:
                            utils.save_checkpoint(opt, model, infos, optimizer, append='best')

                        break

                eval_done = False

                # 设置学习参数
                if epoch_done:
                    # Transformer相关
                    if not opt.noamopt and not opt.reduce_on_plateau:
                        if epoch > opt.learning_rate_decay_start >= 0:
                            frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                            decay_factor = opt.learning_rate_decay_rate ** frac
                            opt.current_lr = opt.learning_rate_base * decay_factor
                        else:
                            opt.current_lr = opt.learning_rate_base
                        utils.set_lr(optimizer, opt.current_lr)

                    # scheduled sampling
                    if epoch > opt.scheduled_sampling_start >= 0:
                        frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                        opt.ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
                        model.ss_prob = opt.ss_prob

                    # SCST
                    if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                        sc_flag = True
                        init_scorer(opt.cached_tokens)
                    else:
                        sc_flag = False

                    # 结构损失
                    if opt.structure_after != -1 and epoch >= opt.structure_after:
                        struc_flag = True
                        init_scorer(opt.cached_tokens)
                    else:
                        struc_flag = False

                    epoch_done = False

                # start = time.time()
                # Transformer Warmup
                if opt.use_warmup and (iteration < opt.noamopt_warmup):
                    opt.current_lr = opt.learning_rate_base * (iteration + 1) / opt.noamopt_warmup
                    utils.set_lr(optimizer, opt.current_lr)

                data = loader.get_batch('base_train')
                # print('\r Read data:', time.time() - start, end="")

                torch.cuda.synchronize()
                start = time.time()

                tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
                tmp = [_ if _ is None else _.cuda() for _ in tmp]
                fc_feats, att_feats, labels, masks, att_masks = tmp

                optimizer.zero_grad()
                model_out = dp_lw_model(fc_feats, att_feats, labels, masks, att_masks, data['gts'], torch.arange(0, len(data['gts'])), sc_flag, struc_flag)

                loss = model_out['loss'].mean()

                loss.backward()

                # 梯度截断
                if opt.grad_clip_value != 0:
                    getattr(torch.nn.utils, 'clip_grad_{}_'.format(opt.grad_clip_mode))(model.parameters(), opt.grad_clip_value)

                optimizer.step()

                train_loss = loss.item()
                torch.cuda.synchronize()
                end = time.time()

                # 输出
                if struc_flag:
                    print('Base Training:', "iter {} (epoch {}), train_loss = {:.3f}, lm_loss = {:.3f}, struc_loss = {:.3f}, time/batch = {:.3f}" \
                          .format(iteration, epoch, train_loss, model_out['lm_loss'].mean().item(), model_out['struc_loss'].mean().item(), end - start))
                elif not sc_flag:
                    print('Base Training:', "iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                          .format(iteration, epoch, train_loss, end - start))
                else:
                    print('Base Training:', "iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                          .format(iteration, epoch, model_out['reward'].mean(), end - start))

                # 更新迭代计数器，如果到达epoch边界，需要调整一些参数
                iteration += 1
                if data['bounds']['wrapped']:
                    epoch += 1
                    epoch_done = True

                # 将训练结构写入到日志中
                if iteration % opt.losses_log_every == 0:
                    tb_summary_writer.add_scalar('train_loss', train_loss, iteration)
                    if opt.noamopt:
                        opt.current_lr = optimizer.rate()
                    elif opt.reduce_on_plateau:
                        opt.current_lr = optimizer.current_lr
                    tb_summary_writer.add_scalar('learning_rate', opt.current_lr, iteration)
                    tb_summary_writer.add_scalar('scheduled_sampling_prob', model.ss_prob, iteration)
                    if sc_flag:
                        tb_summary_writer.add_scalar('avg_reward', model_out['reward'].mean(), iteration)
                    elif struc_flag:
                        tb_summary_writer.add_scalar('lm_loss', model_out['lm_loss'].mean().item(), iteration)
                        tb_summary_writer.add_scalar('struc_loss', model_out['struc_loss'].mean().item(), iteration)
                        tb_summary_writer.add_scalar('reward', model_out['reward'].mean().item(), iteration)
                        tb_summary_writer.add_scalar('reward_var', model_out['reward'].var(1).mean(), iteration)

                    histories['loss_history'][iteration] = train_loss if not sc_flag else model_out['reward'].mean()
                    histories['lr_history'][iteration] = opt.current_lr
                    histories['ss_prob_history'][iteration] = model.ss_prob

                # 信息更新
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['loader_state_dict'] = loader.state_dict()

                # 根据需要，在两个模式下评估模型
                if (iteration % opt.save_checkpoint_every == 0 and not opt.save_every_epoch) or (epoch_done and opt.save_every_epoch):
                    eval_kwargs = {'split': 'base_val', 'dataset': opt.input_json}
                    eval_kwargs.update(vars(opt))
                    val_loss, predictions, lang_stats, _ = eval_utils.eval_split(dp_model, lw_model.crit, loader, eval_kwargs)

                    if opt.reduce_on_plateau:
                        if 'CIDEr' in lang_stats:
                            optimizer.scheduler_step(-lang_stats['CIDEr'])
                        else:
                            optimizer.scheduler_step(val_loss)

                    # 将评估结果写入日志
                    tb_summary_writer.add_scalar('validation loss', val_loss, iteration)
                    if lang_stats is not None:
                        for k, v in lang_stats.items():
                            tb_summary_writer.add_scalar(k, v, iteration)

                    histories['val_result_history'][iteration] = {'loss': val_loss, 'lang_stats': lang_stats, 'predictions': predictions}

                    # 根据CIDEr指标选择最佳模型
                    if opt.language_eval == 1:
                        current_score = lang_stats['CIDEr']
                    else:
                        current_score = - val_loss

                    best_flag = False

                    if best_val_score is None or current_score > best_val_score:
                        best_val_score = current_score
                        best_flag = True

                    infos['best_val_score'] = best_val_score

                    utils.save_checkpoint(opt, model, infos, optimizer, histories)

                    if opt.save_history_ckpt:
                        utils.save_checkpoint(opt, model, infos, optimizer, append=str(epoch) if opt.save_every_epoch else str(iteration))

                    if best_flag:
                        utils.save_checkpoint(opt, model, infos, optimizer, append='best')

                    eval_done = True

        except (RuntimeError, KeyboardInterrupt):
            print('Save ckpt on exception ...')
            utils.save_checkpoint(opt, model, infos, optimizer)
            print('Save ckpt done.')
            stack_trace = traceback.format_exc()
            print(stack_trace)

        infos['stage'] = 2

    # dummy配置下，不进行微调
    if opt.train_only == 0:
        # 微调训练
        infos['stage'] = 2
        epoch_done = True
        loader.reset_iterator('support')

        # 加载最佳模型，如果中断位置在第二阶段，则不进行模型加载
        if opt.start_from and infos['stage_saved'] == 2:
            pass
        else:
            # 否则加载stage 1的最佳模型进行微调
            print('Finetuning:', "loading best model from stage 1")
            model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model_best' + '.pth')))
            optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer_best' + '.pth')))

            lw_model = LossWrapper(model, opt)
            # 多GPU封装
            dp_model = torch.nn.DataParallel(model)
            dp_model.vocab = getattr(model, 'vocab', None)
            dp_lw_model = torch.nn.DataParallel(lw_model)

        model.set_stage(infos['stage'])
        infos['stage_saved'] = 2

        # 冻结除了最后一个logit层之外的所有参数
        for name, parameter in dp_lw_model.module.named_parameters():
            if 'logit' not in name:
                parameter.requires_grad = False
            else:
                parameter.requires_grad = True

        # 因为计数器没有清零，所以这里是直接加上去
        max_epochs_all = opt.max_epochs_base + opt.max_epochs_finetune

        try:
            while True:
                # 达到最大epoch限制，跳出
                if epoch >= max_epochs_all != -2:
                    utils.save_checkpoint(opt, model, infos, optimizer, histories, append='finetune')
                    break

                # 设置学习参数
                if epoch_done:
                    # Transformer相关
                    if not opt.noamopt and not opt.reduce_on_plateau:
                        if epoch > opt.learning_rate_decay_start_finetune + opt.max_epochs_base >= 0:
                            frac = (epoch - opt.learning_rate_decay_start_finetune - opt.max_epochs_base) // opt.learning_rate_decay_every_finetune
                            decay_factor = opt.learning_rate_decay_rate_finetune ** frac
                            opt.current_lr = opt.learning_rate_finetune * decay_factor
                        else:
                            opt.current_lr = opt.learning_rate_finetune

                        utils.set_lr(optimizer, opt.current_lr)

                    # scheduled sampling
                    if epoch > opt.scheduled_sampling_start_finetune + opt.max_epochs_base >= 0:
                        frac = (epoch - opt.scheduled_sampling_start_finetune - opt.max_epochs_base) // opt.scheduled_sampling_increase_every_finetune
                        opt.ss_prob = min(opt.scheduled_sampling_increase_prob_finetune * frac, opt.scheduled_sampling_max_prob_finetune)
                        model.ss_prob = opt.ss_prob

                    # SCST
                    if opt.self_critical_after != -1 and epoch >= opt.self_critical_after:
                        sc_flag = True
                        init_scorer(opt.cached_tokens)
                    else:
                        sc_flag = False

                    # 结构损失
                    if opt.structure_after != -1 and epoch >= opt.structure_after:
                        struc_flag = True
                        init_scorer(opt.cached_tokens)
                    else:
                        struc_flag = False

                    epoch_done = False

                # start = time.time()
                # Transformer Warmup
                # if opt.use_warmup and (iteration < opt.noamopt_warmup):
                #     opt.current_lr = opt.learning_rate * (iteration + 1) / opt.noamopt_warmup
                #     utils.set_lr(optimizer, opt.current_lr)

                data = loader.get_batch('support')

                torch.cuda.synchronize()
                start = time.time()

                tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
                tmp = [_ if _ is None else _.cuda() for _ in tmp]
                fc_feats, att_feats, labels, masks, att_masks = tmp

                optimizer.zero_grad()
                model_out = dp_lw_model(fc_feats, att_feats, labels, masks, att_masks, data['gts'], torch.arange(0, len(data['gts'])), sc_flag, struc_flag)

                loss = model_out['loss'].mean()

                loss.backward()

                # 梯度截断
                if opt.grad_clip_value != 0:
                    getattr(torch.nn.utils, 'clip_grad_{}_'.format(opt.grad_clip_mode))(model.parameters(), opt.grad_clip_value)

                optimizer.step()

                train_loss = loss.item()
                torch.cuda.synchronize()
                end = time.time()

                # 输出
                if struc_flag:
                    print('Finetuning:', "iter {} (epoch {}), train_loss = {:.3f}, lm_loss = {:.3f}, struc_loss = {:.3f}, time/batch = {:.3f}" \
                          .format(iteration, epoch, train_loss, model_out['lm_loss'].mean().item(), model_out['struc_loss'].mean().item(), end - start))
                elif not sc_flag:
                    print('Finetuning:', "iter {} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                          .format(iteration, epoch, train_loss, end - start))
                else:
                    print('Finetuning:', "iter {} (epoch {}), avg_reward = {:.3f}, time/batch = {:.3f}" \
                          .format(iteration, epoch, model_out['reward'].mean(), end - start))

                # 更新迭代计数器，如果到达epoch边界，需要调整一些参数
                iteration += 1
                if data['bounds']['wrapped']:
                    epoch += 1
                    epoch_done = True

                # 将训练结构写入到日志中
                if iteration % opt.losses_log_every == 0:
                    tb_summary_writer.add_scalar('train_loss', train_loss, iteration)
                    if opt.noamopt:
                        opt.current_lr = optimizer.rate()
                    elif opt.reduce_on_plateau:
                        opt.current_lr = optimizer.current_lr
                    tb_summary_writer.add_scalar('learning_rate', opt.current_lr, iteration)
                    tb_summary_writer.add_scalar('scheduled_sampling_prob', model.ss_prob, iteration)
                    if sc_flag:
                        tb_summary_writer.add_scalar('avg_reward', model_out['reward'].mean(), iteration)
                    elif struc_flag:
                        tb_summary_writer.add_scalar('lm_loss', model_out['lm_loss'].mean().item(), iteration)
                        tb_summary_writer.add_scalar('struc_loss', model_out['struc_loss'].mean().item(), iteration)
                        tb_summary_writer.add_scalar('reward', model_out['reward'].mean().item(), iteration)
                        tb_summary_writer.add_scalar('reward_var', model_out['reward'].var(1).mean(), iteration)

                    histories['loss_history'][iteration] = train_loss if not sc_flag else model_out['reward'].mean()
                    histories['lr_history'][iteration] = opt.current_lr
                    histories['ss_prob_history'][iteration] = model.ss_prob

                # 信息更新
                infos['iter'] = iteration
                infos['epoch'] = epoch
                infos['loader_state_dict'] = loader.state_dict()

                if (iteration % opt.save_checkpoint_every == 0 and not opt.save_every_epoch) or (epoch_done and opt.save_every_epoch):
                    utils.save_checkpoint(opt, model, infos, optimizer, histories, append='finetune')

                    if opt.save_history_ckpt:
                        utils.save_checkpoint(opt, model, infos, optimizer, append=str(epoch) if opt.save_every_epoch else str(iteration))

        except (RuntimeError, KeyboardInterrupt):
            print('Save ckpt on exception ...')
            utils.save_checkpoint(opt, model, infos, optimizer)
            print('Save ckpt done.')
            stack_trace = traceback.format_exc()
            print(stack_trace)


opt = opts.parse_opt()
train(opt)
