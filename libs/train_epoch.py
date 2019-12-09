import torch
from torch.autograd import Variable
import time
import os
import sys
import numpy as np
from libs.utils import AverageMeter, calculate_accuracy, calculate_accuracy_pt_0_4

import pdb

def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger, tb_writer=None):
    print('train at epoch {}'.format(epoch))

    model.train()

    # pytroch version check
    torch_version = float(torch.__version__[:3])

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()    
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
            
        if not opt.no_cuda:
            targets = targets.cuda(async=True)
        if opt.model == 'vgg':
            inputs = inputs.squeeze()
        inputs = Variable(inputs)
        targets = Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if torch_version < 0.4:
            acc = calculate_accuracy(outputs, targets)
            losses.update(loss.data[0], inputs.size(0))
        else:
            acc = calculate_accuracy_pt_0_4(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        print('Train Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  acc=accuracies))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })

    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss/total', losses.avg, epoch)
        tb_writer.add_scalar('train/acc/top1_acc_action', accuracies.avg, epoch)
        tb_writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)

    return batch_time.sum, data_time.sum


def train_adv_epoch(epoch, data_loader, model, criterions, optimizer, opt,
                epoch_logger, batch_logger, warm_up_epochs, tb_writer=None):
    print('train at epoch {}'.format(epoch))

    model.train()

    # pytroch version check
    torch_version = float(torch.__version__[:3])

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    act_losses = AverageMeter()
    place_losses = AverageMeter()
    if opt.is_place_entropy and epoch>=opt.warm_up_epochs:
        place_entropy_losses = AverageMeter()
    act_accuracies = AverageMeter()
    place_accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets, places) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        if not opt.no_cuda:
            targets = targets.cuda(async=True)
            places = places.cuda(async=True)
        if opt.model == 'vgg':
            inputs = inputs.squeeze()
        inputs = Variable(inputs)
        targets = Variable(targets)

        if opt.is_place_adv:
            outputs, outputs_places = model(inputs)
        else:
            outputs = model(inputs)
        loss_act = criterions['action_cross_entropy'](outputs, targets)
        
        if opt.is_place_adv:
            loss_place = criterions['places_cross_entropy'](outputs_places, places)
            if opt.is_place_entropy and epoch>=opt.warm_up_epochs:
                loss_place_entropy = criterions['places_entropy'](outputs_places)
                loss = loss_act + loss_place + opt.weight_entropy_loss*loss_place_entropy
            else:
                loss = loss_act + loss_place
        else:
            if opt.is_place_entropy and epoch>=opt.warm_up_epochs:
                loss_place_entropy = criterions['places_entropy'](outputs_places)
                loss = loss_act + opt.weight_entropy_loss*loss_place_entropy
            else:
                loss = loss_act

        if torch_version < 0.4:
            acc = calculate_accuracy(outputs, targets)
            losses.update(loss.data[0], inputs.size(0))
        else:
            act_acc = calculate_accuracy_pt_0_4(outputs, targets)
            if opt.is_place_adv:
                if opt.is_place_soft:                
                    _, places_hard_target = torch.max(places, 1)
                    place_acc = calculate_accuracy_pt_0_4(outputs_places, places_hard_target)
                else:
                    place_acc = calculate_accuracy_pt_0_4(outputs_places, places)
                place_losses.update(loss_place.item(), inputs.size(0))    
            else:
                place_losses.update(0, inputs.size(0))

            losses.update(loss.item(), inputs.size(0))
            act_losses.update(loss_act.item(), inputs.size(0))            
            if opt.is_place_entropy and epoch>=opt.warm_up_epochs:
                place_entropy_losses.update(loss_place_entropy.item(), inputs.size(0))
        act_accuracies.update(act_acc, inputs.size(0))
        if opt.is_place_adv:
            place_accuracies.update(place_acc, inputs.size(0))
        else:
            place_accuracies.update(0, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss total': losses.val,
            'loss act'  : act_losses.val,
            'loss place': place_losses.val,
            'acc act': act_accuracies.val,
            'acc place': place_accuracies.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        print('Train Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\n'
              'Total Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Action Loss {loss_act.val:.4f} ({loss_act.avg:.4f})\t'
              'Place Loss {loss_place.val:.4f} ({loss_place.avg:.4f})\n'
              'Acc action {act_acc.val:.3f} ({act_acc.avg:.3f})\t'
              'Acc place {place_acc.val:.3f} ({place_acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  loss_act=act_losses,
                  loss_place=place_losses,
                  act_acc=act_accuracies,
                  place_acc=place_accuracies))

    epoch_logger.log({
        'epoch': epoch,
        'loss total': losses.avg,
        'loss act': act_losses.avg,
        'loss place': place_losses.avg,
        'acc act': act_accuracies.avg,
        'acc place': place_accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })
    
    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)
    
    if tb_writer is not None:
        tb_writer.add_scalar('train/loss/total', losses.avg, epoch)
        tb_writer.add_scalar('train/loss/action', act_losses.avg, epoch)
        tb_writer.add_scalar('train/loss/place', place_losses.avg, epoch)
        if opt.is_place_entropy and epoch>=opt.warm_up_epochs:
            tb_writer.add_scalar('train/loss/place_entropy', place_entropy_losses.avg, epoch)
        tb_writer.add_scalar('train/acc/top1_acc_action', act_accuracies.avg, epoch)
        tb_writer.add_scalar('train/acc/top1_acc_place', place_accuracies.avg, epoch)
        if epoch < warm_up_epochs:
            tb_writer.add_scalar('train/lr_new', optimizer.param_groups[0]['lr'], epoch)
        else:
            tb_writer.add_scalar('train/lr_pretrained', optimizer.param_groups[0]['lr'], epoch)
            tb_writer.add_scalar('train/lr_new', optimizer.param_groups[-1]['lr'], epoch)


def train_adv_msk_epoch(epoch, data_loaders, model, criterions, optimizer, opt,
                epoch_logger, batch_logger, warm_up_epochs, tb_writer=None):
    print('train at epoch {}'.format(epoch))

    model.train()

    # pytroch version check
    torch_version = float(torch.__version__[:3])

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    act_losses = AverageMeter()
    msk_act_losses = AverageMeter()
    place_losses = AverageMeter()
    if opt.is_place_entropy and epoch>=opt.warm_up_epochs:
        place_entropy_losses = AverageMeter()
    act_accuracies = AverageMeter()
    msk_act_accuracies = AverageMeter()
    place_accuracies = AverageMeter()

    data_loader_unmasked = data_loaders[0]
    data_loader_masked = data_loaders[1]

    end_time = time.time()
    for i, ((inputs_unmasked, targets_unmasked, places_unmasked), (inputs_masked, targets_masked, maskings)) in enumerate(zip(data_loader_unmasked, data_loader_masked)):    
        data_time.update(time.time() - end_time)
        if not opt.no_cuda:
            targets_unmasked = targets_unmasked.cuda(async=True)
            places_unmasked = places_unmasked.cuda(async=True)
            targets_masked = targets_masked.cuda(async=True)

        # # ------------------------------------------------
        # #  Train using Action CE loss and Place ADV loss
        # # ------------------------------------------------    
        if opt.model == 'vgg':
            inputs_unmasked = inputs_unmasked.squeeze()    
        inputs_unmasked = Variable(inputs_unmasked)
        targets_unmasked = Variable(targets_unmasked)

        if opt.is_mask_entropy:
            if opt.is_place_adv:
                if opt.is_mask_adv:
                    outputs_unmasked, outputs_places_unmasked, outputs_rev_unmasked = model (inputs_unmasked)
                else:
                    outputs_unmasked, outputs_places_unmasked = model(inputs_unmasked)
            else:
                if opt.is_mask_adv:
                    outputs_unmasked, outputs_rev_unmasked = model(inputs_unmasked)
                else:
                    outputs_unmasked = model(inputs_unmasked)
        elif opt.is_mask_cross_entropy:
            if opt.is_place_adv:
                outputs_unmasked, outputs_places_unmasked, outputs_rev_unmasked = model (inputs_unmasked)
            else:
                outputs_unmasked, outputs_rev_unmasked = model(inputs_unmasked)

        loss_act = criterions['action_cross_entropy'](outputs_unmasked, targets_unmasked)
        if opt.is_place_adv:
            loss_place = criterions['places_cross_entropy'](outputs_places_unmasked, places_unmasked)
            if opt.is_place_entropy and epoch>=opt.warm_up_epochs:
                loss_place_entropy = criterions['places_entropy'](outputs_places_unmasked)
                loss = loss_act + loss_place + opt.weight_entropy_loss*loss_place_entropy
            else:
                loss = loss_act + loss_place
        else:
            if opt.is_place_entropy and epoch>=opt.warm_up_epochs:
                loss_place_entropy = criterions['places_entropy'](outputs_places_unmasked)
                loss = loss_act + opt.weight_entropy_loss*loss_place_entropy
            else:
                loss = loss_act

        if torch_version < 0.4:
            acc = calculate_accuracy(outputs_unmasked, targets_unmasked)
            losses.update(loss.data[0], inputs_unmasked.size(0))
        else:
            act_acc = calculate_accuracy_pt_0_4(outputs_unmasked, targets_unmasked)
            if opt.is_place_adv:
                if opt.is_place_soft:                
                    _, places_hard_target = torch.max(places_unmasked, 1)
                    place_acc = calculate_accuracy_pt_0_4(outputs_places_unmasked, places_hard_target)
                else:
                    place_acc = calculate_accuracy_pt_0_4(outputs_places_unmasked, places_unmasked)            
                place_losses.update(loss_place.item(), inputs_unmasked.size(0))
            else:                
                place_losses.update(0, inputs_unmasked.size(0))
                            
            losses.update(loss.item(), inputs_unmasked.size(0))
            act_losses.update(loss_act.item(), inputs_unmasked.size(0))            
            if opt.is_place_entropy and epoch>=opt.warm_up_epochs:
                place_entropy_losses.update(loss_place_entropy.item(), inputs_unmasked.size(0))
        act_accuracies.update(act_acc, inputs_unmasked.size(0))
        if opt.is_place_adv:
            place_accuracies.update(place_acc, inputs_unmasked.size(0))
        else:
            place_accuracies.update(0, inputs_unmasked.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # -------------------------------------------------
        #  Train using Mask Action Entropy loss (maximize)
        # -------------------------------------------------
        print('num of actual masking_inds = {}/{}'.format(torch.sum(maskings), maskings.shape[0]))

        if opt.model == 'vgg':
            inputs_masked = inputs_masked.squeeze()            
        inputs_masked = Variable(inputs_masked)
        targets_masked = Variable(targets_masked)
        
        if opt.is_mask_entropy:
            if opt.is_place_adv:
                if opt.is_mask_adv:
                    outputs_masked, outputs_places_masked, outputs_rev_masked = model(inputs_masked)
                else:
                    outputs_masked, outputs_places_masked = model(inputs_masked)
            else:
                if opt.is_mask_adv:
                    outputs_masked, outputs_rev_masked = model(inputs_masked)
                else:
                    outputs_masked = model(inputs_masked)
        elif opt.is_mask_cross_entropy:
            if opt.is_place_adv:
                outputs_masked, outputs_places_masked, outputs_rev_masked = model(inputs_masked)
            else:
                outputs_masked, outputs_rev_masked = model(inputs_masked)

        if opt.is_mask_entropy:
            if opt.is_mask_adv:
                loss_action_entropy = criterions['mask_criterion'](outputs_rev_masked)
            else:
                loss_action_entropy = criterions['mask_criterion'](outputs_masked)
            msk_loss = loss_action_entropy
        elif opt.is_mask_cross_entropy:
            loss_action_cross_entropy = criterions['mask_criterion'](outputs_rev_masked, targets_masked)
            msk_loss = loss_action_cross_entropy

        if torch_version < 0.4:
            if opt.is_mask_adv:
                acc = calculate_accuracy(outputs_rev_masked, targets_masked)           
            else:
                acc = calculate_accuracy(outputs_masked, targets_masked)
        else:
            if opt.is_mask_adv:
                msk_act_acc = calculate_accuracy_pt_0_4(outputs_rev_masked, targets_masked)    
            else:
                msk_act_acc = calculate_accuracy_pt_0_4(outputs_masked, targets_masked)                  
        msk_act_losses.update(msk_loss.item(), inputs_masked.size(0))
        msk_act_accuracies.update(msk_act_acc, inputs_masked.size(0))        

        optimizer.zero_grad()
        msk_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader_unmasked) + (i + 1),
            'loss total': losses.val,
            'loss act'  : act_losses.val,
            'loss place': place_losses.val,
            'acc act': act_accuracies.val,
            'acc place': place_accuracies.val,
            'lr': optimizer.param_groups[0]['lr']
        })

        print('Train Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Total Loss {loss.val:.4f} ({loss.avg:.4f})\n'
              'Action Loss {loss_act.val:.4f} ({loss_act.avg:.4f})\t'
              'Place Loss {loss_place.val:.4f} ({loss_place.avg:.4f})\t'
              'Mask Action Confusion Loss {msk_loss.val:.4f} ({msk_loss.avg:.4f})\n'
              'Acc Action {act_acc.val:.3f} ({act_acc.avg:.3f})\t'
              'Acc Place {place_acc.val:.3f} ({place_acc.avg:.3f})\t'
              'Acc Mask Action {msk_act_acc.val:.3f} ({msk_act_acc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader_unmasked),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  loss_act=act_losses,
                  loss_place=place_losses,
                  msk_loss=msk_act_losses,
                  act_acc=act_accuracies,
                  place_acc=place_accuracies,
                  msk_act_acc=msk_act_accuracies
                  ))

    epoch_logger.log({
        'epoch': epoch,
        'loss total': losses.avg,
        'loss act': act_losses.avg,
        'loss place': place_losses.avg,
        'acc act': act_accuracies.avg,
        'acc place': place_accuracies.avg,
        'lr': optimizer.param_groups[0]['lr']
    })
    
    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'save_{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, save_file_path)

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss/total', losses.avg, epoch)
        tb_writer.add_scalar('train/loss/action', act_losses.avg, epoch)
        tb_writer.add_scalar('train/loss/place', place_losses.avg, epoch)
        if opt.is_place_entropy and epoch>=opt.warm_up_epochs:
            tb_writer.add_scalar('train/loss/place_entropy', place_entropy_losses.avg, epoch)
        tb_writer.add_scalar('train/acc/top1_acc_action', act_accuracies.avg, epoch)
        tb_writer.add_scalar('train/acc/top1_acc_place', place_accuracies.avg, epoch)
        if epoch < warm_up_epochs:
            tb_writer.add_scalar('train/lr_new', optimizer.param_groups[0]['lr'], epoch)
        else:
            tb_writer.add_scalar('train/lr_pretrained', optimizer.param_groups[0]['lr'], epoch)
            tb_writer.add_scalar('train/lr_new', optimizer.param_groups[-1]['lr'], epoch)
        tb_writer.add_scalar('train/msk_loss/msk_action', msk_act_losses.avg, epoch)        
        tb_writer.add_scalar('train/msk_acc/top1_msk_acc_action', msk_act_accuracies.avg, epoch)
