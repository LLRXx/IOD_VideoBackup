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


def _residual_scale_values(model, suffix):
    """Return scalar residual-scale parameters whose names end with suffix."""
    values = []
    for name, parameter in model.named_parameters():
        if name.endswith(suffix):
            values.append((name, float(parameter.detach().cpu().item())))
    return values


def _print_epoch_summary(epoch, train_metrics, model):
    """Print epoch losses and the current RGAM/SCAM residual scales."""
    metrics = ' '.join(
        '{}={:.6f}'.format(name, value)
        for name, value in train_metrics.items())
    rgam_scales = _residual_scale_values(model, 'rgam.residual_scale')
    scam_scales = _residual_scale_values(model, 'scam.residual_scale')
    rgam_text = ', '.join(
        '{}={:.6f}'.format(name, value) for name, value in rgam_scales) or 'none'
    scam_text = ', '.join(
        '{}={:.6f}'.format(name, value) for name, value in scam_scales) or 'none'
    print('[Epoch {}] train_metrics: {} | RGAM alpha: {} | SCAM alpha: {}'.format(
        epoch, metrics, rgam_text, scam_text), flush=True)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def worker_init_fn(dump):
    set_seed(GLOBAL_SEED)

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
        use_cpca=opt.use_cpca,
        cpca_reduction=opt.cpca_reduction,
        cpca_kernel_sizes=opt.cpca_kernel_sizes,
        cpca_residual_scale=opt.cpca_residual_scale,
        use_scam=opt.use_scam,
        scam_reduction=opt.scam_reduction,
        scam_spatial_kernel=opt.scam_spatial_kernel,
        scam_channel_group=opt.scam_channel_group,
        scam_residual_scale=opt.scam_residual_scale,
        scam_only=opt.scam_only,
        use_rgam=opt.use_rgam,
        rgam_groups=opt.rgam_groups,
        rgam_reduction_c=opt.rgam_reduction_c,
        rgam_reduction_s=opt.rgam_reduction_s,
        rgam_spatial_size=(opt.resize_height // 16, opt.resize_width // 16),
        rgam_residual_scale=opt.rgam_residual_scale)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    start_epoch = opt.start_epoch

    #load from the imagenet pre-trained model
    if opt.pretrain_model == 'imagenet':
        model = load_imagenet_pretrained_model(opt, model)
    else:
        print("there is no pretrained model, init from random parameter.")

    #load from the already trained model
    if opt.load_model != '':
        if opt.load_model_weights_only or opt.scam_only:
            model = load_model(model, opt.load_model)
        else:
            model, optimizer, _, _ = load_model(model, opt.load_model, optimizer, opt.lr)

    if opt.scam_only:
        model.freeze_scam_only()
        optimizer = torch.optim.Adam(
            (parameter for parameter in model.parameters()
             if parameter.requires_grad), opt.lr)

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
        _print_epoch_summary(epoch, log_dict_train, model)

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
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            logger.write('Drop LR to ' + str(lr) + '\n')
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    logger.close()


if __name__ == '__main__':
    os.system("rm -rf tmp")
    opt = opts().parse()
    main(opt)
