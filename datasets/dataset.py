from datasets.kinetics import Kinetics, Kinetics_adv, Kinetics_bkgmsk, Kinetics_human_msk, Kinetics_adv_msk
from datasets.activitynet import ActivityNet
from datasets.ucf101 import UCF101
from datasets.hmdb51 import HMDB51
from datasets.diving48 import Diving48


def get_training_set(opt, spatial_transform, temporal_transform,
                     target_transform):
    assert opt.dataset in ['kinetics', 'kinetics_adv', 'kinetics_bkgmsk', 'kinetics_adv_msk', 'activitynet', 'ucf101', 'hmdb51', 'diving48']

    if opt.dataset == 'kinetics':
        training_data = Kinetics(
            opt.video_path+'/train',
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'kinetics_adv':
        training_data = Kinetics_adv(
            opt.video_path+'/train',
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            place_pred_path=opt.place_pred_path,
            is_place_soft_label=opt.is_place_soft)      
    elif opt.dataset == 'kinetics_bkgmsk':
        training_data = Kinetics_bkgmsk(
            opt.video_path+'/train',
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            detection_path=opt.human_dets_path,
            mask_ratio=opt.mask_ratio)            
    elif opt.dataset == 'kinetics_adv_msk':    
        training_data_1 = Kinetics_adv(
            opt.video_path+'/train',
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            place_pred_path=opt.place_pred_path,
            is_place_soft_label=opt.is_place_soft)   
        training_data_2 = Kinetics_human_msk(
            opt.video_path+'/train',
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            detection_path=opt.human_dets_path,
            mask_ratio=opt.mask_ratio)         
        training_data = [training_data_1, training_data_2]
    elif opt.dataset == 'activitynet':
        training_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            'training',
            False,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'ucf101':
        training_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'hmdb51':
        training_data = HMDB51(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)
    elif opt.dataset == 'diving48':
        training_data = Diving48(
            opt.video_path,
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    return training_data


def get_validation_set(opt, spatial_transform, temporal_transform,
                       target_transform):
    assert opt.dataset in ['kinetics', 'kinetics_adv', 'kinetics_bkgmsk', 'kinetics_human_msk', 'kinetics_adv_msk', 'activitynet', 'ucf101', 'hmdb51', 'diving48']

    if opt.dataset == 'kinetics':
        validation_data = Kinetics(
            opt.video_path+'/val',
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'kinetics_adv':
        validation_data = Kinetics_adv(
            opt.video_path+'/val',
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration,
            place_pred_path=opt.place_pred_path,
            is_place_soft_label=opt.is_place_soft)      
    elif opt.dataset == 'kinetics_bkgmsk':
        validation_data = Kinetics_bkgmsk(
            opt.video_path+'/val',
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration,
            detection_path=opt.human_dets_path,
            mask_ratio=opt.mask_ratio)                  
    elif opt.dataset == 'kinetics_adv_msk':
        validation_data_1 = Kinetics_adv(
            opt.video_path+'/val',
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration,
            place_pred_path=opt.place_pred_path,
            is_place_soft_label=opt.is_place_soft)     
        validation_data_2 = Kinetics_human_msk(
            opt.video_path+'/val',
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration,
            detection_path=opt.human_dets_path,
            mask_ratio=opt.mask_ratio)                   
        validation_data = [validation_data_1, validation_data_2]
    elif opt.dataset == 'activitynet':
        validation_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            'validation',
            False,
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ucf101':
        validation_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration,
            vis=opt.vis)
    elif opt.dataset == 'hmdb51':
        validation_data = HMDB51(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration,
            vis=opt.vis)
    elif opt.dataset == 'diving48':
        validation_data = Diving48(
            opt.video_path,
            opt.annotation_path,
            'validation',
            opt.n_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration,
            vis=opt.vis)            
    return validation_data


def get_test_set(opt, spatial_transform, temporal_transform, target_transform):
    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51', 'diving48']
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if opt.dataset == 'kinetics':
        test_data = Kinetics(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'activitynet':
        test_data = ActivityNet(
            opt.video_path,
            opt.annotation_path,
            subset,
            True,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'ucf101':
        test_data = UCF101(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'hmdb51':
        test_data = HMDB51(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)
    elif opt.dataset == 'diving48':
        test_data = Diving48(
            opt.video_path,
            opt.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=opt.sample_duration)

    return test_data
