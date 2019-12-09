import torch
import torch.utils.data as data
from PIL import Image, ImageDraw, ImageStat
import os
import math
import functools
import json
import copy
import numpy as np
import glob as gb

from libs.utils import load_value_file
import pdb

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            if subset == 'testing':
                video_names.append('test/{}'.format(key))
            else:
                label = value['annotations']['label']
                video_names.append('{}/{}'.format(label, key))
                annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            continue

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i][:-14].split('/')[1]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class

def is_video_valid_det(cur_vid_dets, n_frames, det_th=0.3, ratio_th=0.7):
    cnt = 0
    for i,v in cur_vid_dets.items():
        if np.sum(v['human_boxes'][:,-1] > det_th) > 0:
            cnt += 1

    if cnt/n_frames >= ratio_th:
        return True
    else:
        return False        

def make_dataset_human_det(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration, dets):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    num_samples_wo_filtering = 0
    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            continue

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        if n_samples_for_each_video == 1:
            num_samples_wo_filtering += 1
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            num_samples_wo_filtering += np.arange(1, n_frames, step).shape[0]

        cur_cls = video_path.split('/')[-2]
        vid = video_path.split('/')[-1][:11]
        if not is_video_valid_det(dets[cur_cls][vid], n_frames, det_th=0.3, ratio_th=0.7):
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i][:-14].split('/')[1]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    print('len(dataset) after filtering the videos without sufficient detections: [{}/{}]'.format(len(dataset), num_samples_wo_filtering))
    
    # repeat the dataset so that the number of samples can be matched to the original dataset w/o filtering
    if len(dataset) < num_samples_wo_filtering:
        num_repeat = np.ceil(num_samples_wo_filtering/len(dataset)).astype(np.int32)
        dataset_repeat = copy.deepcopy(dataset)
        for i in range(1, num_repeat):           
            dataset_shuffled = copy.deepcopy(dataset) 
            np.random.shuffle(dataset_shuffled)
            dataset_repeat += dataset_shuffled

    return dataset_repeat[:num_samples_wo_filtering], idx_to_class


def gen_mask(img, dets, idx, th=0.3):
    mask_img = Image.new('L', (img.width, img.height), 255)
    cnt = 0
    for det in dets:
        if det[-1] >= th:
            cnt += 1
            poly = [(det[0], det[1]), (det[0], det[3]), (det[2], det[3]), (det[2], det[1])]
            ImageDraw.Draw(mask_img).polygon(poly, fill=0) # black-out mask
    mask = mask_img
    
    is_eff_masking = True if cnt > 0 else False
    
    return mask, is_eff_masking


def maskout_human(img, dets, idx, th=0.3):
    pixel_mean = tuple(np.round(np.mean(img,axis=(0,1))).astype(np.int32))
    cnt = 0
    for det in dets:
        if det[-1] >= th:
            cnt += 1
            poly = [(det[0], det[1]), (det[0], det[3]), (det[2], det[3]), (det[2], det[1])]
            ImageDraw.Draw(img).polygon(poly, fill=pixel_mean) # black-out mask
    
    is_eff_masking = True if cnt > 0 else False
  
    return img, is_eff_masking


class Kinetics(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)


class Kinetics_adv(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader,
                 place_pred_path=None,
                 is_place_soft_label=False):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.is_place_soft_label = is_place_soft_label
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform        
        if place_pred_path is not None:
            if 'train' in subset:
                place_pred_files = gb.glob(place_pred_path+'/*{}*.npy'.format('train'))
            elif 'val' in subset:
                place_pred_files = gb.glob(place_pred_path+'/*{}*.npy'.format('val'))
            self.place_pred = dict()
            for place_pred_file in place_pred_files:
                self.place_pred.update(np.load(place_pred_file).item())       
                
        vid = list(self.place_pred[ list(self.place_pred.keys())[0]].keys())[0]
        if isinstance(self.place_pred[list(self.place_pred.keys())[0]][vid]['pred_cls'], np.ndarray):
            self.multiple_place_inds = True
        else:
            self.multiple_place_inds = False
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, place) where target is class_index of the target class, and place is the place_index of the video
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.is_place_soft_label:
            if self.multiple_place_inds:
                place_soft_target = np.mean(self.place_pred[path.split('/')[-2]][self.data[index]['video_id']]['probs'],axis=0)
            else:
                place_soft_target = self.place_pred[path.split('/')[-2]][self.data[index]['video_id']]['probs']
            return clip, target, place_soft_target
        else:          
            place_index = self.place_pred[path.split('/')[-2]][self.data[index]['video_id']]['pred_cls']
            if self.multiple_place_inds:
                place_index = np.bincount(place_index).argmax()
            return clip, target, place_index

    def __len__(self):
        return len(self.data)


class Kinetics_bkgmsk(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader,
                 detection_path=None,
                 mask_ratio=0.5):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)
        self.subset= subset

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform        
        if detection_path is not None:
            if 'train' in subset:
                detection_file = os.path.join(detection_path, 'detection_{}_merged_rearranged.npy'.format('train'))
                print('loading human dets from {} ...'.format(detection_file))       
                self.human_dets_tr = np.load(detection_file).item()
                print('loading human dets from {} done'.format(detection_file))       
            elif 'val' in subset:
                detection_file = os.path.join(detection_path, 'detection_{}_merged_rearranged.npy'.format('val'))
                print('loading human dets from {} ...'.format(detection_file))       
                self.human_dets_val = np.load(detection_file).item()
                print('loading human dets from {} done'.format(detection_file))       
            
        self.loader = get_loader()
        self.mask_ratio = mask_ratio # blurred bkg video ratio within batch

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, place) where target is class_index of the target class, and place is the place_index of the video
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        
        # get human detections for the current clip
        human_dets = self.human_dets_tr if self.subset == 'training' else self.human_dets_val

        cur_human_dets = []
        for idx in frame_indices:
            cur_human_dets.append(human_dets[path.split('/')[-2]][path.split('/')[-1][:11]][idx]['human_boxes'])

        # mask out the backgrounds 
        fg_imgs, masks = [], []
        rands = np.random.rand(1)
        for i,frm in enumerate(clip):
            bkg_img = Image.new('L', (frm.width, frm.height), 0)
            mask = bkg_img
            if cur_human_dets[i].shape[0] == 0:
                fg_img = frm
            else:
                if rands < self.mask_ratio:
                    mask, is_eff_masking = gen_mask(frm, cur_human_dets[i], i)
                    fg_img = Image.composite(bkg_img, frm, mask)
                else:
                    fg_img = frm
            fg_imgs.extend([fg_img])
            masks.extend([mask])
        
        clip = fg_imgs

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)


class Kinetics_human_msk(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader,
                 detection_path=None,
                 mask_ratio=0.5,
                 mask_th=0.5):
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform        

        if detection_path is not None:
            if 'train' in subset:
                detection_file = os.path.join(detection_path, 'detection_{}_merged_rearranged.npy'.format('train'))
                print('loading human dets from {} ...'.format(detection_file))       
                self.human_dets = np.load(detection_file).item()
                print('loading human dets from {} done'.format(detection_file))       
            elif 'val' in subset:
                detection_file = os.path.join(detection_path, 'detection_{}_merged_rearranged.npy'.format('val'))
                print('loading human dets from {} ...'.format(detection_file))       
                self.human_dets = np.load(detection_file).item()
                print('loading human dets from {} done'.format(detection_file))     
        
        self.loader = get_loader()
        self.mask_ratio = mask_ratio # blurred bkg video ratio within batch
        self.mask_th = mask_th

        self.data, self.class_names = make_dataset_human_det(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration, self.human_dets)
        self.subset= subset

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)

        # get human detections for the current clip
        cur_human_dets = []
        for idx in frame_indices:
            cur_human_dets.append(self.human_dets[path.split('/')[-2]][path.split('/')[-1][:11]][idx]['human_boxes'])

        # mask out the backgrounds 
        fg_imgs, masks = [], []
        rands = np.random.rand(1)                
        mask_cnt = 0
        
        for i,frm in enumerate(clip):
            bkg_img = Image.new('L', (frm.width, frm.height), 0)
            mask = bkg_img
            if cur_human_dets[i].shape[0] == 0:
                fg_img = frm
            else:
                if rands < self.mask_ratio:
                    fg_img, cnt = maskout_human(frm, cur_human_dets[i], i)
                    mask_cnt += cnt
                else:
                    fg_img = frm            
            fg_imgs.extend([fg_img])
            masks.extend([mask])
        
        clip = fg_imgs
        is_masking = True if mask_cnt/len(clip) >= self.mask_th else False

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
                
        return clip, target, is_masking

    def __len__(self):
        return len(self.data)


class Kinetics_adv_msk(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader,
                 place_pred_path=None,
                 is_place_soft_label=False,
                 detection_path=None,
                 mask_ratio=0.5,
                 mask_th=0.5):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)
        self.subset= subset

        self.is_place_soft_label = is_place_soft_label
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform        
        if place_pred_path is not None:
            if 'train' in subset:
                place_pred_files = gb.glob(place_pred_path+'/*{}*.npy'.format('train'))
            elif 'val' in subset:
                place_pred_files = gb.glob(place_pred_path+'/*{}*.npy'.format('val'))
            self.place_pred = dict()
            for place_pred_file in place_pred_files:
                self.place_pred.update(np.load(place_pred_file).item())       
                
        vid = list(self.place_pred[ list(self.place_pred.keys())[0]].keys())[0]
        if isinstance(self.place_pred[list(self.place_pred.keys())[0]][vid]['pred_cls'], np.ndarray):
            self.multiple_place_inds = True
        else:
            self.multiple_place_inds = False

        if detection_path is not None:
            if 'train' in subset:
                detection_file = os.path.join(detection_path, 'detection_{}_merged_rearranged.npy'.format('train'))
                print('loading human dets from {} ...'.format(detection_file))       
                self.human_dets_tr = np.load(detection_file).item()
                print('loading human dets from {} done'.format(detection_file))       
            elif 'val' in subset:
                detection_file = os.path.join(detection_path, 'detection_{}_merged_rearranged.npy'.format('val'))
                print('loading human dets from {} ...'.format(detection_file))       
                self.human_dets_val = np.load(detection_file).item()
                print('loading human dets from {} done'.format(detection_file))     
        
        self.loader = get_loader()
        self.mask_ratio = mask_ratio # blurred bkg video ratio within batch
        self.mask_th = mask_th

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, place) where target is class_index of the target class, and place is the place_index of the video
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)

        # get human detections for the current clip
        human_dets = self.human_dets_tr if self.subset == 'training' else self.human_dets_val

        cur_human_dets = []
        for idx in frame_indices:
            cur_human_dets.append(human_dets[path.split('/')[-2]][path.split('/')[-1][:11]][idx]['human_boxes'])

        # mask out the backgrounds 
        fg_imgs, masks = [], []
        rands = np.random.rand(1)                
        mask_cnt = 0
        
        for i,frm in enumerate(clip):
            bkg_img = Image.new('L', (frm.width, frm.height), 0)
            mask = bkg_img
            if cur_human_dets[i].shape[0] == 0:
                fg_img = frm
            else:
                if rands < self.mask_ratio:
                    fg_img, cnt = maskout_human(frm, cur_human_dets[i], i)
                    mask_cnt += cnt
                else:
                    fg_img = frm            
            fg_imgs.extend([fg_img])
            masks.extend([mask])
        
        clip = fg_imgs
        is_masking = True if mask_cnt/len(clip) >= self.mask_th else False

        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.is_place_soft_label:
            if self.multiple_place_inds:
                place_soft_target = np.mean(self.place_pred[path.split('/')[-2]][self.data[index]['video_id']]['probs'],axis=0)
            else:
                place_soft_target = self.place_pred[path.split('/')[-2]][self.data[index]['video_id']]['probs']
            return clip, target, place_soft_target, is_masking
        else:          
            place_index = self.place_pred[path.split('/')[-2]][self.data[index]['video_id']]['pred_cls']
            if self.multiple_place_inds:
                place_index = np.bincount(place_index).argmax()
            return clip, target, place_index, is_masking

    def __len__(self):
        return len(self.data)
