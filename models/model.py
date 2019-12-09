import torch
from torch import nn

from models import resnet, pre_act_resnet, wide_resnet, resnext, densenet, vgg
import pdb

def generate_model(opt):
    assert opt.model in [
        'resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet', 'vgg'
    ]

    if opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        from models.resnet import get_fine_tuning_parameters, get_adv_fine_tuning_parameters

        if opt.model_depth == 10:
            model = resnet.resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                is_adv=opt.is_place_adv,
                is_human_mask_adv=opt.is_mask_adv,
                alpha=opt.alpha,
                alpha_hm=opt.alpha_hm,
                num_places_classes=opt.num_places_classes,
                num_place_hidden_layers=opt.num_place_hidden_layers,
                num_human_mask_adv_hidden_layers=opt.num_human_mask_adv_hidden_layers)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                is_adv=opt.is_place_adv,
                is_human_mask_adv=opt.is_mask_adv,
                alpha=opt.alpha,
                alpha_hm=opt.alpha_hm,
                num_places_classes=opt.num_places_classes,
                num_place_hidden_layers=opt.num_place_hidden_layers,
                num_human_mask_adv_hidden_layers=opt.num_human_mask_adv_hidden_layers)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                is_adv=opt.is_place_adv,
                is_human_mask_adv=opt.is_mask_adv,
                alpha=opt.alpha,
                alpha_hm=opt.alpha_hm,
                num_places_classes=opt.num_places_classes,
                num_place_hidden_layers=opt.num_place_hidden_layers,
                num_human_mask_adv_hidden_layers=opt.num_human_mask_adv_hidden_layers)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                is_adv=opt.is_place_adv,
                is_human_mask_adv=opt.is_mask_adv,
                alpha=opt.alpha,
                alpha_hm=opt.alpha_hm,
                num_places_classes=opt.num_places_classes,
                num_place_hidden_layers=opt.num_place_hidden_layers,
                num_human_mask_adv_hidden_layers=opt.num_human_mask_adv_hidden_layers)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                is_adv=opt.is_place_adv,
                is_human_mask_adv=opt.is_mask_adv,
                alpha=opt.alpha,
                alpha_hm=opt.alpha_hm,
                num_places_classes=opt.num_places_classes,
                num_place_hidden_layers=opt.num_place_hidden_layers,
                num_human_mask_adv_hidden_layers=opt.num_human_mask_adv_hidden_layers)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                is_adv=opt.is_place_adv,
                is_human_mask_adv=opt.is_mask_adv,
                alpha=opt.alpha,
                alpha_hm=opt.alpha_hm,
                num_places_classes=opt.num_places_classes,
                num_place_hidden_layers=opt.num_place_hidden_layers,
                num_human_mask_adv_hidden_layers=opt.num_human_mask_adv_hidden_layers)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                is_adv=opt.is_place_adv,
                is_human_mask_adv=opt.is_mask_adv,
                alpha=opt.alpha,
                alpha_hm=opt.alpha_hm,
                num_places_classes=opt.num_places_classes,
                num_place_hidden_layers=opt.num_place_hidden_layers,
                num_human_mask_adv_hidden_layers=opt.num_human_mask_adv_hidden_layers)
    elif opt.model == 'wideresnet':
        assert opt.model_depth in [50]

        from models.wide_resnet import get_fine_tuning_parameters

        if opt.model_depth == 50:
            model = wide_resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                k=opt.wide_resnet_k,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'resnext':
        assert opt.model_depth in [50, 101, 152]

        from models.resnext import get_fine_tuning_parameters

        if opt.model_depth == 50:
            model = resnext.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnext.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnext.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'preresnet':
        assert opt.model_depth in [18, 34, 50, 101, 152, 200]

        from models.pre_act_resnet import get_fine_tuning_parameters

        if opt.model_depth == 18:
            model = pre_act_resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = pre_act_resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = pre_act_resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = pre_act_resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = pre_act_resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 200:
            model = pre_act_resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'densenet':
        assert opt.model_depth in [121, 169, 201, 264]

        from models.densenet import get_fine_tuning_parameters

        if opt.model_depth == 121:
            model = densenet.densenet121(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 169:
            model = densenet.densenet169(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 201:
            model = densenet.densenet201(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 264:
            model = densenet.densenet264(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    elif opt.model == 'vgg':
        
        from models.vgg import get_fine_tuning_parameters, get_adv_fine_tuning_parameters
        
        model = vgg.build_vgg(
            num_classes=opt.n_classes,
            is_adv=opt.is_place_adv,
            is_human_mask_adv=opt.is_mask_adv,
            alpha=opt.alpha,
            alpha_hm=opt.alpha_hm,
            num_places_classes=opt.num_places_classes,
            num_place_hidden_layers=opt.num_place_hidden_layers,
            num_human_mask_adv_hidden_layers=opt.num_human_mask_adv_hidden_layers)

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)

        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            
            if opt.model != 'vgg':
                assert opt.arch == pretrain['arch']
            # else:
            #     pdb.set_trace()
            # pdb.set_trace()

            # model.load_state_dict(pretrain['state_dict'])
            model_dict = model.state_dict()

            # 1. filter out unnecessary keys and the last fc layers' weights
            pretrained_dict = dict()
            if 'state_dict' in pretrain:
                for k,v in pretrain['state_dict'].items():
                    if ((k in model_dict) and (v.shape == model_dict[k].shape)):
                        pretrained_dict[k] = v
            else:
                for k,v in pretrain.items():
                    new_k = 'module.vgg.'+ k
                    if ((new_k in model_dict) and (v.shape == model_dict[new_k].shape)):
                        pretrained_dict[new_k] = v
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)

            if not opt.not_replace_last_fc:
                if opt.model == 'densenet':
                    model.module.classifier = nn.Linear(
                        model.module.classifier.in_features, opt.n_finetune_classes)
                    model.module.classifier = model.module.classifier.cuda()
                else:
                    model.module.fc = nn.Linear(model.module.fc.in_features,
                                                opt.n_finetune_classes)
                    model.module.fc = model.module.fc.cuda()

            if opt.is_place_adv or opt.is_mask_cross_entropy or opt.is_mask_entropy:
                # pdb.set_trace()
                parameters = get_adv_fine_tuning_parameters(model, opt.ft_begin_index, opt.new_layer_lr, not_replace_last_fc=opt.not_replace_last_fc, is_human_mask_adv=opt.is_mask_adv, slower_place_mlp=opt.slower_place_mlp, slower_hm_mlp=opt.slower_hm_mlp)
            else:
                parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)

            return model, parameters
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            pretrain = torch.load(opt.pretrain_path)
            
            if opt.model != 'vgg':
                assert opt.arch == pretrain['arch']
            # else:
            #     pdb.set_trace()

            # model.load_state_dict(pretrain['state_dict'])
            model_dict = model.state_dict()

            # 1. filter out unnecessary keys and the last fc layers' weights
            pretrained_dict = dict()
            if 'state_dict' in pretrain:
                for k,v in pretrain['state_dict'].items():
                    if ((k in model_dict) and (v.shape == model_dict[k].shape)):
                        pretrained_dict[k] = v
            else:
                for k,v in pretrain.items():
                    new_k = 'module.vgg.'+ k
                    if ((new_k in model_dict) and (v.shape == model_dict[new_k].shape)):
                        pretrained_dict[new_k] = v
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)

            if not opt.not_replace_last_fc:
                if opt.model == 'densenet':
                    model.classifier = nn.Linear(
                        model.classifier.in_features, opt.n_finetune_classes)
                else:
                    model.fc = nn.Linear(model.fc.in_features,
                                                opt.n_finetune_classes)
                                                
            if opt.is_place_adv:
                parameters = get_adv_fine_tuning_parameters(model, opt.ft_begin_index, opt.new_layer_lr, not_replace_last_fc=opt.not_replace_last_fc, is_human_mask_adv=opt.is_mask_adv, slower_place_mlp=opt.slower_place_mlp, slower_hm_mlp=opt.slower_hm_mlp)
            else:
                parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters

    return model, model.parameters()
