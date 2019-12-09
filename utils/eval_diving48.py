import json

import numpy as np
import pandas as pd
import sys, os
sys.path.append('/home/jinchoi/src/3D-ResNets-PyTorch')
from opts import parse_opts
import pdb

class Diving48classification(object):
    def __init__(self, ground_truth_filename=None, class_def_filename=None, prediction_filename=None,
                 subset='validation', verbose=False, top_k=1):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.subset = subset
        self.verbose = verbose
        self.top_k = top_k
        self.ap = None
        self.hit_at_k = None
        # Import ground truth and predictions.
        self.ground_truth = self._import_ground_truth(
            ground_truth_filename)
        self.activity_index = self._get_class_labels(class_def_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            print ('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.ground_truth)
            print ('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print ('\tNumber of predictions: {}'.format(nr_pred))

    def _get_class_labels(self, data_file_path):
        with open(data_file_path, 'r') as data_file:
            data = json.load(data_file)
        data = ['_'.join(row) for row in data]
        class_labels_map = {}
        index = 0
        for class_label in data:
            class_labels_map[class_label] = index
            index += 1
        return class_labels_map
    
    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        
        # pdb.set_trace()
        # Checking format
        # if not all([field in data.keys() for field in self.gt_fields]):
            # raise IOError('Please input a valid ground truth file.')

        # Initialize data frame
        video_lst, label_lst = [], []
        for cur_data in data:
            video_lst.append(cur_data['vid_name'])
            label_lst.append(cur_data['label'])
        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     'label': label_lst})
        ground_truth = ground_truth.drop_duplicates().reset_index(drop=True)

        return ground_truth

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        # if not all([field in data.keys() for field in self.pred_fields]):
            # raise IOError('Please input a valid prediction file.')

        # Initialize data frame
        video_lst, label_lst, score_lst = [], [], []
        for videoid, v in data['results'].items():
            for result in v:
                label = self.activity_index[result['label']]
                video_lst.append(videoid)
                label_lst.append(label)
                score_lst.append(result['score'])
        prediction = pd.DataFrame({'video-id': video_lst,
                                   'label': label_lst,
                                   'score': score_lst})
        return prediction

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        hit_at_k = compute_video_hit_at_k(self.ground_truth,
                                          self.prediction, top_k=self.top_k)
        if self.verbose:
            print ('[RESULTS] Performance on Diving48 video '
                   'classification task.')
            print ('\tError@{}: {}'.format(self.top_k, 1.0 - hit_at_k))
            #print '\tAvg Hit@{}: {}'.format(self.top_k, avg_hit_at_k)
        self.hit_at_k = hit_at_k

################################################################################
# Metrics
################################################################################
def compute_video_hit_at_k(ground_truth, prediction, top_k=3):
    """Compute accuracy at k prediction between ground truth and
    predictions data frames. This code is greatly inspired by evaluation
    performed in Karpathy et al. CVPR14.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 'label']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 'label', 'score']

    Outputs
    -------
    acc : float
        Top k accuracy score.
    """
    video_ids = np.unique(ground_truth['video-id'].values)
    avg_hits_per_vid = np.zeros(video_ids.size)
    for i, vid in enumerate(video_ids):
        pred_idx = prediction['video-id'] == vid
        if not pred_idx.any():
            continue
        this_pred = prediction.loc[pred_idx].reset_index(drop=True)
        # Get top K predictions sorted by decreasing score.
        sort_idx = this_pred['score'].values.argsort()[::-1][:top_k]
        this_pred = this_pred.loc[sort_idx].reset_index(drop=True)
        # Get labels and compare against ground truth.
        pred_label = this_pred['label'].tolist()
        gt_idx = ground_truth['video-id'] == vid
        gt_label = ground_truth.loc[gt_idx]['label'].tolist()
        avg_hits_per_vid[i] = np.mean([1 if this_label in pred_label else 0
                                       for this_label in gt_label])
    return float(avg_hits_per_vid.mean())


if __name__ == '__main__':
    opt = parse_opts()
    opt.class_def_path = os.path.join(opt.root_path, opt.annotation_path, 'Diving48_vocab.json')

    opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path, 'Diving48_test.json')
    
    # pdb.set_trace()
    diving48_classification = Diving48classification(opt.annotation_path, opt.class_def_path, opt.prediction_path, subset='val', verbose=True, top_k=1)
    diving48_classification.evaluate()
    print(diving48_classification.hit_at_k)
