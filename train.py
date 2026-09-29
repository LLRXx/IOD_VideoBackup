from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import torch
import torch.utils.data
from opts import opts
from utils.model import create_model, load_model, save_model, load_imagenet_pretrained_model
from trainer.logger import Logger
from datasets.init_dataset import get_dataset
from trainer.trainer import Trainer
import numpy as np
import random
import tensorboardX


GLOBAL_SEED = 317

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def worker_init_fn(dump):
    set_seed(GLOBAL_SEED)


def configure_dpdf_only_training(model):
    """Freeze the baseline and expose only the inserted DPDF to Adam.

    Frozen BatchNorm layers are kept in eval mode because ``requires_grad``
    does not stop running-statistics updates.  This keeps the official
    baseline feature statistics fixed during the DPDF-only ablation.
    """
    trainable_names = []
    for name, parameter in model.named_parameters():
        # ``STA_Framework`` registers the block at the root as
        # ``dpdf.<parameter>``.  Checking for ``.dpdf.`` alone misses that
        # valid root-level spelling and leaves the optimizer with no params.
        name_parts = name.split('.')
        parameter.requires_grad = 'dpdf' in name_parts
        if parameter.requires_grad:
            trainable_names.append(name)

    if not trainable_names:
        raise RuntimeError(
            '--train_dpdf_only was requested, but no DPDF parameters were found. '
            'Make sure --use_dpdf is enabled.')

    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    print('DPDF-only training enabled; trainable parameter count: {}'.format(
        len(trainable_names)))
    print('DPDF-only training enabled; trainable parameters:')
    for name in trainable_names:
        print('  ' + name)
    return trainable_names


def create_optimizer(model, opt):
    """Create optimizer groups for joint fine-tuning.

    The inserted DPDF is allowed to move faster than the pretrained backbone,
    while the fusion/detection head uses an intermediate rate.  Parameter
    names are grouped by the model modules registered by STA_Framework.
    """
    lr_backbone = opt.lr if opt.lr_backbone < 0 else opt.lr_backbone
    lr_dpdf = opt.lr if opt.lr_dpdf < 0 else opt.lr_dpdf
    lr_head = opt.lr if opt.lr_head < 0 else opt.lr_head

    grouped = {
        'backbone': {'params': [], 'lr': lr_backbone},
        'dpdf': {'params': [], 'lr': lr_dpdf},
        'head': {'params': [], 'lr': lr_head},
    }
    grouped_names = {key: [] for key in grouped}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.split('.')[0] == 'dpdf':
            group_name = 'dpdf'
        elif name.split('.')[0] == 'backbone':
            group_name = 'backbone'
        else:
            group_name = 'head'
        grouped[group_name]['params'].append(parameter)
        grouped_names[group_name].append(name)

    param_groups = []
    for group_name in ('backbone', 'dpdf', 'head'):
        group = grouped[group_name]
        if group['params']:
            group['name'] = group_name
            group['initial_lr'] = group['lr']
            param_groups.append(group)
            print('Optimizer group {}: {} parameters, lr={}'.format(
                group_name, len(grouped_names[group_name]), group['lr']))

    if not param_groups:
        raise RuntimeError('No trainable parameters were found for the optimizer.')
    return torch.optim.Adam(param_groups)


def main(opt):
    set_seed(opt.seed)
    torch.backends.cudnn.benchmark = True
    print('dataset: ' + opt.dataset + '   task:  ' + opt.task)
    Dataset = get_dataset(opt.dataset)
    opt = opts().update_dataset(opt, Dataset)

    #log
    train_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'train'))
    epoch_train_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'train_epoch'))
    val_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'val'))
    epoch_val_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'val_epoch'))

    logger = Logger(opt, epoch_train_writer, epoch_val_writer)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    #model define
    model = create_model(
        opt.arch,
        opt.branch_info,
        opt.head_conv,
        opt.K,
        use_dpdf=opt.use_dpdf,
        dpdf_heads=opt.dpdf_heads,
        dpdf_branch_channels=opt.dpdf_branch_channels,
        dpdf_deform_kernel=opt.dpdf_deform_kernel,
        dpdf_dilation_rates=opt.dpdf_dilation_rates,
        dpdf_temporal=opt.dpdf_temporal,
        dpdf_temporal_align=opt.dpdf_temporal_align,
        dpdf_dynamic_dilation=opt.dpdf_dynamic_dilation,
        dpdf_consistency=opt.dpdf_consistency,
    )
    optimizer = None
    start_epoch = opt.start_epoch

    #load from the imagenet pre-trained model
    if opt.pretrain_model == 'imagenet':
        model = load_imagenet_pretrained_model(opt, model)
    else:
        print("there is no pretrained model, init from random parameter.")

    #load from the already trained model
    if opt.load_model != '':
        if opt.load_model_weights_only or opt.train_dpdf_only:
            model = load_model(model, opt.load_model)
        else:
            optimizer = torch.optim.Adam(model.parameters(), opt.lr)
            model, optimizer, _, _ = load_model(model, opt.load_model, optimizer, opt.lr)

    if opt.train_dpdf_only:
        configure_dpdf_only_training(model)

    if optimizer is None:
        if opt.train_dpdf_only:
            optimizer_parameters = (
                parameter for parameter in model.parameters()
                if parameter.requires_grad)
            optimizer = torch.optim.Adam(optimizer_parameters, opt.lr_dpdf if opt.lr_dpdf > 0 else opt.lr)
        else:
            optimizer = create_optimizer(model, opt)

    #Trainer Class
    trainer = Trainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

    print('training...')
    print('GPU allocate:', opt.chunk_sizes)

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        print('epoch is ', epoch)

        #Dataset Class
        Dataset = get_dataset(opt.dataset)
        opt = opts().update_dataset(opt, Dataset)

        #DataLoader
        train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        drop_last=True,
        worker_init_fn=worker_init_fn
        ) 
        val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        drop_last=True,
        worker_init_fn=worker_init_fn
        )  

        # train
        log_dict_train = trainer.train(epoch, train_loader, train_writer)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('epcho/{}'.format(k), v, epoch, 'train')
            logger.write('train: {} {:8f} | '.format(k, v))
        logger.write('\n')

        # save the model
        if opt.save_all:
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            model_name = 'model_[{}]_{}.pth'.format(epoch, time_str)
            save_model(os.path.join(opt.save_dir, model_name),
                       model, optimizer, epoch, log_dict_train['loss'])
        else:
            model_name = 'model_last.pth'
            save_model(os.path.join(opt.save_dir, model_name),
                       model, optimizer, epoch, log_dict_train['loss'])

        # evaluate the model
        if opt.val_epoch:
            with torch.no_grad():
                log_dict_val = trainer.val(epoch, val_loader, val_writer)
            for k, v in log_dict_val.items():
                logger.scalar_summary('epcho/{}'.format(k), v, epoch, 'val')
                logger.write('val: {} {:8f} | '.format(k, v))
        logger.write('\n')

        #decrese the learning rate
        if epoch in opt.lr_step:
            logger.write('Drop optimizer learning rates by 0.1\n')
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

    logger.close()


if __name__ == '__main__':
    os.system("rm -rf tmp")
    opt = opts().parse()
    main(opt)
