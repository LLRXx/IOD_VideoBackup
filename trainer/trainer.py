from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from .losses import FocalLoss, RegL1Loss,STAloss,STAloss2

from progress.bar import Bar
from utils.data_parallel import DataParallel
from utils.gate_monitor import GateStatistics
from utils.utils import AverageMeter


class ModleWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModleWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        oracle_gate = batch.get('oracle_gate')
        if oracle_gate is None:
            [output] = self.model(batch['input'])
        else:
            [output] = self.model(
                batch['input'], oracle_gate=oracle_gate)
        loss, loss_stats = self.loss(output, batch)
        return output, loss, loss_stats

class TrainLoss(torch.nn.Module):
    def __init__(self, opt):
        super(TrainLoss, self).__init__()
        self.crit_hm = FocalLoss()
        self.crit_mov = RegL1Loss()
        self.crit_wh = RegL1Loss()
        self.crit_STAloss = STAloss(opt)
        self.crit_STAloss2 = STAloss2(opt)
        self.opt = opt

    def forward(self, output, batch):
        opt = self.opt
        output['hm'] = torch.clamp(output['hm'].sigmoid_(), min=1e-4, max=1 - 1e-4)

        hm_loss = self.crit_hm(output['hm'], batch['hm'])

        mov_loss = self.crit_mov(output['mov'], batch['mask'],
                                 batch['index'], batch['mov'])

        wh_loss = self.crit_wh(output['wh'], batch['mask'],
                               batch['index'], batch['wh'],
                               index_all=batch['index_all'])

        sta_sin_loss,sta_cos_loss = self.crit_STAloss(batch['centerKpoints'], batch['wh'],
                                output['hm'],output['wh'],output['STA_offset'],
                                batch['mask'],batch['index'],
                               index_all=batch['index_all'])

        # sta_sin_loss2,sta_cos_loss2 = self.crit_STAloss2(batch['centerKpoints'], batch['wh'],
        #                                               output['hm'],output['wh'],output['STA_offset'],
        #                                               batch['mask'],batch['index'],
        #                                               index_all=batch['index_all'])

        if self.opt.loss_option == 'STAloss':
            loss = opt.hm_weight * hm_loss.mean() + opt.wh_weight * wh_loss.mean() + opt.sta_weight * (sta_sin_loss.mean()+sta_cos_loss.mean())
        else:
            loss = opt.hm_weight * hm_loss.mean() + opt.wh_weight * wh_loss.mean() + opt.mov_weight * mov_loss.mean()

        loss = loss.unsqueeze(0)
        hm_loss = hm_loss.unsqueeze(0)
        wh_loss = wh_loss.unsqueeze(0)
        mov_loss = mov_loss.unsqueeze(0)
        sta_sin_loss = sta_sin_loss.mean().unsqueeze(0)
        sta_cos_loss = sta_cos_loss.mean().unsqueeze(0)
        # sta_sin_loss2 = sta_sin_loss2.mean().unsqueeze(0)
        # sta_cos_loss2 = sta_cos_loss2.mean().unsqueeze(0)
        # print(sta_cos_loss.detach().cpu().numpy(),"==",sta_cos_loss2.detach().cpu().numpy())
        # print(sta_sin_loss.detach().cpu().numpy(),"==",sta_sin_loss2.detach().cpu().numpy())

        loss_stats = {'loss': loss, 'hm_loss': hm_loss,'wh_loss': wh_loss,'mov_loss':mov_loss,'sta_sin_loss':sta_sin_loss,'sta_cos_loss':sta_cos_loss}

        return loss, loss_stats


class Trainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats = ['loss', 'hm_loss',  'wh_loss','sta_sin_loss','sta_cos_loss'] if self.opt.loss_option == 'STAloss'\
                    else  ['loss', 'hm_loss',  'wh_loss','mov_loss']
        self.model_with_loss = ModleWithLoss(model, TrainLoss(opt))

    def train(self, epoch, data_loader, writer):
        return self.run_epoch('train', epoch, data_loader, writer)

    def val(self, epoch, data_loader, writer):
        return self.run_epoch('val', epoch, data_loader, writer)

    @staticmethod
    def _gradient_norm(parameters):
        squared_norm = None
        for parameter in parameters:
            if parameter.grad is None:
                continue
            value = parameter.grad.detach().float().pow(2).sum()
            squared_norm = value if squared_norm is None else squared_norm + value
        if squared_norm is None:
            return 0.0
        return squared_norm.sqrt().item()

    @staticmethod
    def _unwrap_network(model_with_loss):
        wrapped = (model_with_loss.module
                   if hasattr(model_with_loss, 'module') else model_with_loss)
        return wrapped.model

    def _cpca_gradient_stats(self, model_with_loss):
        network = self._unwrap_network(model_with_loss)
        if not (getattr(network, 'use_cpca_gate', False)
                or getattr(network, 'use_cpca_oracle_gate', False)):
            return {}
        cpca_norm = self._gradient_norm(network.cpca.parameters())
        backbone_norm = self._gradient_norm(network.backbone.parameters())
        return {
            'cpca_grad_norm': cpca_norm,
            'backbone_grad_norm': backbone_norm,
            'cpca_to_backbone_grad_ratio': (
                cpca_norm / max(backbone_norm, 1e-12)),
        }

    def run_epoch(self, phase, epoch, data_loader, writer):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
            network = self._unwrap_network(model_with_loss)
            if (getattr(network, 'use_cpca_oracle_gate', False)
                    and not getattr(self.opt, 'unfreeze_oracle_baseline', False)):
                # Keep all baseline BatchNorm/dropout behavior fixed while
                # optimizing the Oracle CPCA adapter only.
                network.backbone.eval()
                network.branch.eval()
                network.R2D.eval()
                network.cpca.train()
        else:
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        gate_statistics = GateStatistics()
        avg_gradient_stats = {
            name: AverageMeter() for name in (
                'cpca_grad_norm',
                'backbone_grad_norm',
                'cpca_to_backbone_grad_ratio')
        }
        num_iters = len(data_loader)
        bar = Bar(opt.exp_id, max=num_iters)
        monitor_interval = max(1, opt.visual_per_inter)

        for iter, batch in enumerate(data_loader):
            if iter >= num_iters:
                break

            for k in batch:
                if k == 'input':
                    for i in range(len(batch[k])):
                        batch[k][i] = batch[k][i].to(device=opt.device, non_blocking=True)
                else:
                        batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            output, loss, loss_stats = model_with_loss(batch)
            has_gate = gate_statistics.update_from_output(output)

            loss = loss.mean()
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                if iter % monitor_interval == 0 or iter == num_iters - 1:
                    gradient_stats = self._cpca_gradient_stats(model_with_loss)
                    for name, value in gradient_stats.items():
                        avg_gradient_stats[name].update(value)
                self.optimizer.step()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)

            step = (iter // monitor_interval
                    + num_iters // monitor_interval * (epoch - 1))

            for l in self.loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'][0].size(0))

                if phase == 'train' and iter % monitor_interval == 0 and iter != 0:
                    writer.add_scalar('train/{}'.format(l), avg_loss_stats[l].avg, step)
                    writer.flush()
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

            if has_gate:
                batch_gate = output['cpca_gate'].detach().float()
                batch_scale = output['cpca_residual_scale'].detach().float()
                batch_effective = batch_scale.abs() * batch_gate
                Bar.suffix += '|gate {:.4f} |s {:.5f} |eff {:.5f} '.format(
                    batch_gate.mean().item(),
                    batch_scale.mean().item(),
                    batch_effective.mean().item())
                if phase == 'train' and iter % monitor_interval == 0:
                    writer.add_scalar(
                        'train/gate_batch_mean', batch_gate.mean().item(), step)
                    writer.add_scalar(
                        'train/gate_batch_std',
                        batch_gate.std(unbiased=False).item(), step)
                    writer.add_scalar(
                        'train/cpca_residual_scale',
                        batch_scale.mean().item(), step)
                    writer.add_scalar(
                        'train/cpca_effective_strength',
                        batch_effective.mean().item(), step)
                    for name, meter in avg_gradient_stats.items():
                        if meter.count > 0:
                            writer.add_scalar(
                                'train/{}'.format(name), meter.val, step)
                    writer.flush()
            bar.next()
            del output, loss, loss_stats

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret.update(gate_statistics.summary())
        for name, meter in avg_gradient_stats.items():
            if meter.count > 0:
                ret[name] = meter.avg
        ret['time'] = bar.elapsed_td.total_seconds() / 60.
        return ret

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    # MODIFY for pytorch 0.4.0
                    state[k] = v.to(device=device, non_blocking=True)
                    # state[k] = v.to(device=device)
