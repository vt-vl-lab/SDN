import cv2
import os
import sys
import json
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from libs.opts import parse_opts
from models.model import generate_model
from libs.mean import get_mean, get_std
from libs.spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from libs.temporal_transforms import LoopPadding, TemporalRandomCrop, TemporalCenterCrop
from libs.target_transforms import ClassLabel, VideoID
from libs.target_transforms import Compose as TargetCompose
from datasets.dataset import get_training_set, get_validation_set, get_test_set
from libs.utils import Logger, AverageMeter
from libs.train_epoch import train_adv_msk_epoch, train_adv_epoch, train_epoch
from libs.validation_epoch import val_adv_msk_epoch, val_adv_epoch, val_epoch
from libs.test import test
import time
from loss.hloss import HLoss
from loss.soft_cross_entropy import SoftCrossEntropy

import pdb

def main_baseline_model(opt):
    torch.manual_seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)
    
    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(opt.norm_value), norm_method
        ])

        temporal_transform = TemporalRandomCrop(opt.sample_duration)
        target_transform = ClassLabel()
        training_data = get_training_set(opt, spatial_transform,
                                         temporal_transform, target_transform)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
        
        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        
        if opt.dataset in ['kinetics_adv', 'kinetics_bkgmsk', 'kinetics_adv_msk']:
            first_optimizer = optim.SGD(
                parameters[1], # new parameters only
                lr=opt.new_layer_lr,
                momentum=opt.momentum,
                dampening=dampening,
                weight_decay=opt.weight_decay,
                nesterov=opt.nesterov)
            first_scheduler = lr_scheduler.ReduceLROnPlateau(
                first_optimizer, 'min', patience=opt.lr_patience)

            second_optimizer = optim.SGD(
                [
                    {'params': parameters[0]}, # pretrained parameters
                    {'params': parameters[1]}  # new parameters
                ],
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=dampening,
                weight_decay=opt.weight_decay,
                nesterov=opt.nesterov)
            second_scheduler = lr_scheduler.ReduceLROnPlateau(
                second_optimizer, 'min', patience=opt.lr_patience)
        else:
            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=dampening,
                weight_decay=opt.weight_decay,
                nesterov=opt.nesterov)
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=opt.lr_patience)


    if not opt.no_val:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.val_batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])
      
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('run')
    tr_btime_avg = AverageMeter()
    tr_dtime_avg = AverageMeter()
    val_btime_avg = AverageMeter()
    val_dtime_avg = AverageMeter()
    for i in range(opt.begin_epoch, opt.n_epochs + 1):      
        if opt.dataset in ['kinetics_adv', 'kinetics_bkgmsk', 'kinetics_adv_msk']:  
            if i < opt.warm_up_epochs:
                optimizer = first_optimizer
                scheduler = first_scheduler
            else:
                optimizer = second_optimizer
                scheduler = second_scheduler
        if not opt.no_train:
            tr_btime, tr_dtime = train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger)
        if not opt.no_val:
            validation_loss, val_btime, val_dtime = val_epoch(i, val_loader, model, criterion, opt,
                                        val_logger)

        if not opt.no_train and not opt.no_val:
            scheduler.step(validation_loss)
            tr_btime_avg.update(tr_btime)
            tr_dtime_avg.update(tr_dtime)
            val_btime_avg.update(val_btime)
            val_dtime_avg.update(val_dtime)
            print('One epoch tr btime = {:.2f}sec, tr dtime = {:.2f}'.format(tr_btime_avg.avg, tr_dtime_avg.avg))
            print('One epoch val btime = {:.2f}sec, val dtime = {:.2f}'.format(val_btime_avg.avg, val_dtime_avg.avg))
        sys.stdout.flush()

    if opt.test:
        spatial_transform = Compose([
            Scale(int(opt.sample_size / opt.scale_in_test)),
            CornerCrop(opt.sample_size, opt.crop_position_in_test),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = VideoID()

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.val_batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test(test_loader, model, opt, test_data.class_names)

def main_sdn_place_adv_model(opt):
    torch.manual_seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)
    criterion = nn.CrossEntropyLoss() 
    if opt.is_place_soft:
        places_criterion = SoftCrossEntropy()
        print('using soft cross entropy for places')
    else:
        places_criterion = nn.CrossEntropyLoss()
    if opt.is_place_entropy:
        places_entropy_criterion = HLoss(is_maximization=opt.is_entropy_max)
            
    if not opt.no_cuda:
        criterion = criterion.cuda()
        places_criterion = places_criterion.cuda()
        if opt.is_place_entropy:
            places_entropy_criterion = places_entropy_criterion.cuda()
    
    criterions = dict()
    criterions['action_cross_entropy'] = criterion
    criterions['places_cross_entropy'] = places_criterion
    if opt.is_place_entropy:
        criterions['places_entropy'] = places_entropy_criterion

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)
    
    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration)
        target_transform = ClassLabel()
        training_data = get_training_set(opt, spatial_transform,
                                         temporal_transform, target_transform)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True,
            drop_last=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss total', 'loss act', 'loss place', 'acc act', 'acc place', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss total', 'loss act', 'loss place', 'acc act', 'acc place', 'lr'])
        
        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        
        first_optimizer = optim.SGD(
            parameters[1], # new parameters only
            lr=opt.new_layer_lr,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
        first_scheduler = lr_scheduler.ReduceLROnPlateau(
            first_optimizer, 'min', patience=opt.lr_patience)

        second_optimizer = optim.SGD(
            [
                {'params': parameters[0]}, # pretrained parameters
                {'params': parameters[1]}  # new parameters
            ],
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
        second_scheduler = lr_scheduler.ReduceLROnPlateau(
            second_optimizer, 'min', patience=opt.lr_patience)
    
    if not opt.no_val:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=opt.val_batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True,
            drop_last=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss total', 'loss act', 'loss place', 'acc act', 'acc place'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if i < opt.warm_up_epochs:
            optimizer = first_optimizer
            scheduler = first_scheduler
        else:
            optimizer = second_optimizer
            scheduler = second_scheduler
        if not opt.no_train:
            train_adv_epoch(i, train_loader, model, criterions, optimizer, opt,
                        train_logger, train_batch_logger, opt.warm_up_epochs)
        if not opt.no_val:
            validation_loss = val_adv_epoch(i, val_loader, model, criterions, opt, val_logger)

        if not opt.no_train and not opt.no_val:
            scheduler.step(validation_loss)
        sys.stdout.flush()
    
    if opt.test:
        spatial_transform = Compose([
            Scale(int(opt.sample_size / opt.scale_in_test)),
            CornerCrop(opt.sample_size, opt.crop_position_in_test),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = VideoID()

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.val_batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test(test_loader, model, opt, test_data.class_names)

def main_sdn_full_model(opt):
    torch.manual_seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)
    criterion = nn.CrossEntropyLoss() 
    if opt.is_place_soft:
        places_criterion = SoftCrossEntropy()
        print('using soft cross entropy for places')
    else:
        places_criterion = nn.CrossEntropyLoss()
    if opt.is_place_entropy:
        places_entropy_criterion = HLoss(is_maximization=opt.is_entropy_max)
    if opt.is_mask_cross_entropy:
        mask_criterion = nn.CrossEntropyLoss()
    elif opt.is_mask_entropy:
        mask_criterion = HLoss(is_maximization=True)
    
    if not opt.no_cuda:
        criterion = criterion.cuda()
        places_criterion = places_criterion.cuda()
        if opt.is_place_entropy:
            places_entropy_criterion = places_entropy_criterion.cuda()
        mask_criterion = mask_criterion.cuda()
    
    criterions = dict()
    criterions['action_cross_entropy'] = criterion
    criterions['places_cross_entropy'] = places_criterion
    criterions['mask_criterion'] = mask_criterion
    if opt.is_place_entropy:
        criterions['places_entropy'] = places_entropy_criterion    
    
    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)
    
    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration)
        target_transform = ClassLabel()
        training_data = get_training_set(opt, spatial_transform,
                                         temporal_transform, target_transform)
        train_loader_unmasked = torch.utils.data.DataLoader(
            training_data[0],
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=int(opt.n_threads/2),
            pin_memory=True,
            drop_last=True)
        train_loader_masked = torch.utils.data.DataLoader(
            training_data[1],
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=int(opt.n_threads/2),
            pin_memory=True,
            drop_last=True)
        train_loaders = [train_loader_unmasked, train_loader_masked]
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss total', 'loss act', 'loss place', 'acc act', 'acc place', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss total', 'loss act', 'loss place', 'acc act', 'acc place', 'lr'])
        
        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening

        if len(parameters[1])>0:
            first_optimizer = optim.SGD(
                parameters[1], # new parameters only
                lr=opt.new_layer_lr,
                momentum=opt.momentum,
                dampening=dampening,
                weight_decay=opt.weight_decay,
                nesterov=opt.nesterov)
            first_scheduler = lr_scheduler.ReduceLROnPlateau(
                first_optimizer, 'min', patience=opt.lr_patience)

        second_optimizer = optim.SGD(
            [
                {'params': parameters[0]}, # pretrained parameters
                {'params': parameters[1]}  # new parameters
            ],
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
        second_scheduler = lr_scheduler.ReduceLROnPlateau(
            second_optimizer, 'min', patience=opt.lr_patience)
    
    if not opt.no_val:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, target_transform)
        val_loader_unmasked = torch.utils.data.DataLoader(
            validation_data[0],
            batch_size=opt.val_batch_size,
            shuffle=False,
            num_workers=int(opt.n_threads/2),
            pin_memory=True,
            drop_last=True)
        val_loader_masked = torch.utils.data.DataLoader(
            validation_data[1],
            batch_size=opt.val_batch_size,
            shuffle=False,
            num_workers=int(opt.n_threads/2),
            pin_memory=True,
            drop_last=True)
        val_loaders = [val_loader_unmasked, val_loader_masked]
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss total', 'loss act', 'loss place', 'acc act', 'acc place'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if i < opt.warm_up_epochs:
            optimizer = first_optimizer
            scheduler = first_scheduler
        else:
            optimizer = second_optimizer
            scheduler = second_scheduler
        if not opt.no_train:
            if not opt.is_mask_adv:
                if i < opt.warm_up_epochs:
                    train_adv_epoch(i, train_loaders[0], model, criterions, optimizer, opt,
                            train_logger, train_batch_logger, opt.warm_up_epochs)       
                else:
                    train_adv_msk_epoch(i, train_loaders, model, criterions, optimizer, opt,
                                train_logger, train_batch_logger, opt.warm_up_epochs)
            else:
                train_adv_msk_epoch(i, train_loaders, model, criterions, optimizer, opt,
                                train_logger, train_batch_logger, opt.warm_up_epochs)
        if not opt.no_val:
            if not opt.is_mask_adv:
                if i < opt.warm_up_epochs:
                    validation_loss = val_adv_epoch(i, val_loaders[0], model, criterions, opt, val_logger)
                else:
                    validation_loss = val_adv_msk_epoch(i, val_loaders, model, criterions, opt, val_logger)
            else:
                validation_loss = val_adv_msk_epoch(i, val_loaders, model, criterions, opt, val_logger)
        if not opt.no_train and not opt.no_val:
            scheduler.step(validation_loss)
        sys.stdout.flush()
    
    if opt.test:
        spatial_transform = Compose([
            Scale(int(opt.sample_size / opt.scale_in_test)),
            CornerCrop(opt.sample_size, opt.crop_position_in_test),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = VideoID()

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.val_batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test(test_loader, model, opt, test_data.class_names)

if __name__ == '__main__':
    opt = parse_opts()
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if not os.path.exists(opt.result_path):
            os.makedirs(opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value)
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    if (not opt.is_mask_adv) and (not opt.is_place_adv):
        print('training a baseline model')
        main_baseline_model(opt)
    elif opt.is_place_adv and (not opt.is_mask_adv):
        print('training a SDN model with scene adv loss only')
        main_sdn_place_adv_model(opt)
    elif opt.is_place_adv and opt.is_mask_adv:
        print('training a full SDN model')
        main_sdn_full_model(opt)
