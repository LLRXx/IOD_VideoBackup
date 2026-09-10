from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import ast
import json
import os

import torch

from datasets.init_dataset import get_dataset
from opts import opts as project_opts
from utils.gate_monitor import GateStatistics
from utils.model import create_model, load_model


def _read_training_args(opt_file):
    if not os.path.isfile(opt_file):
        return []
    with open(opt_file, 'r') as source:
        lines = source.readlines()
    for index, line in enumerate(lines):
        if line.strip() == '==> Cmd:' and index + 1 < len(lines):
            command = ast.literal_eval(lines[index + 1].strip())
            if command and command[0].endswith('.py'):
                command = command[1:]
            return command
    return []


def _monitor_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Measure the learned VACPCA gate distribution and residual scale '
            'from a saved checkpoint. Unknown arguments are forwarded to '
            'opts.py as model/dataset overrides.'))
    parser.add_argument('--checkpoint', required=True,
                        help='checkpoint produced by train.py')
    parser.add_argument('--opt_file', default=None,
                        help='training opt.txt; defaults to checkpoint folder')
    parser.add_argument('--monitor_split', choices=('train', 'val'),
                        default='val', help='dataset split to measure')
    parser.add_argument('--max_batches', type=int, default=0,
                        help='0 scans the complete split')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='override evaluation batch size')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='override data-loader workers')
    parser.add_argument('--gpus', default=None,
                        help='visible GPUs, or -1 for CPU')
    parser.add_argument('--output', default=None,
                        help='JSON output path')
    return parser


def _build_project_options(monitor_args, project_overrides):
    opt_file = monitor_args.opt_file
    if opt_file is None:
        opt_file = os.path.join(
            os.path.dirname(os.path.abspath(monitor_args.checkpoint)),
            'opt.txt')
    training_args = _read_training_args(opt_file)
    if not training_args:
        print('Warning: no training command found in {}.'.format(opt_file))
        print('Using opts.py defaults plus explicit command-line overrides.')

    opt = project_opts().parse(training_args + project_overrides)
    if monitor_args.batch_size is not None:
        opt.batch_size = monitor_args.batch_size
    if monitor_args.num_workers is not None:
        opt.num_workers = monitor_args.num_workers
    if monitor_args.gpus is not None:
        opt.gpus_str = monitor_args.gpus
        requested_gpus = [int(gpu) for gpu in monitor_args.gpus.split(',')]
        opt.gpus = ([i for i in range(len(requested_gpus))]
                    if requested_gpus[0] >= 0 else [-1])
    return opt, opt_file


def _diagnosis(summary):
    gate_mean = summary['gate_mean']
    gate_std = summary['gate_std']
    scale = abs(summary['cpca_residual_scale_last'])
    if scale < 0.02:
        return ('High collapse risk: residual_scale is close to zero, so the '
                'CPCA branch is nearly disabled globally.')
    if gate_mean < 0.10 or summary['gate_le_0.10_frac'] > 0.90:
        return ('High collapse risk: most gate values are close to zero, so '
                'the CPCA branch is nearly bypassed.')
    if gate_std < 0.02:
        return ('No obvious zero collapse, but the gate varies very little '
                'between clips and has not learned a strong dynamic policy.')
    return ('No obvious gate collapse under these heuristic thresholds; '
            'compare the distribution across later epochs and data subsets.')


def _print_summary(summary):
    ordered_names = (
        'gate_count', 'gate_mean', 'gate_std', 'gate_min', 'gate_p10',
        'gate_p25', 'gate_p50', 'gate_p75', 'gate_p90', 'gate_max',
        'gate_le_0.05_frac', 'gate_le_0.10_frac', 'gate_ge_0.90_frac',
        'gate_ge_0.95_frac', 'cpca_residual_scale_last',
        'cpca_effective_strength_mean', 'cpca_effective_strength_min',
        'cpca_effective_strength_max')
    print('\nGate monitor summary')
    print('-' * 54)
    for name in ordered_names:
        value = summary[name]
        if isinstance(value, int):
            print('{:<36s} {:>16d}'.format(name, value))
        else:
            print('{:<36s} {:>16.8f}'.format(name, value))
    print('-' * 54)
    print(summary['diagnosis'])


def main():
    parser = _monitor_parser()
    monitor_args, project_overrides = parser.parse_known_args()
    checkpoint = os.path.abspath(monitor_args.checkpoint)
    if not os.path.isfile(checkpoint):
        parser.error('checkpoint does not exist: {}'.format(checkpoint))
    if monitor_args.max_batches < 0:
        parser.error('--max_batches must be non-negative')

    opt, opt_file = _build_project_options(
        monitor_args, project_overrides)
    if not (getattr(opt, 'use_cpca_gate', False)
            or getattr(opt, 'use_cpca_oracle_gate', False)):
        parser.error(
            'the reconstructed configuration does not enable a CPCA gate; '
            'check opt.txt or pass --use_cpca with either '
            '--use_cpca_gate or --use_cpca_oracle_gate')

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    use_cuda = opt.gpus[0] >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if opt.gpus[0] >= 0 and not use_cuda:
        print('Warning: CUDA was requested but is unavailable; using CPU.')

    Dataset = get_dataset(opt.dataset)
    opt = project_opts().update_dataset(opt, Dataset)
    model = create_model(
        opt.arch,
        opt.branch_info,
        opt.head_conv,
        opt.K,
        use_cpca=opt.use_cpca,
        cpca_reduction=opt.cpca_reduction,
        cpca_kernel_sizes=opt.cpca_kernel_sizes,
        cpca_residual_scale=opt.cpca_residual_scale,
        use_cpca_gate=opt.use_cpca_gate,
        cpca_gate_hidden=opt.cpca_gate_hidden,
        use_cpca_oracle_gate=opt.use_cpca_oracle_gate)
    model = load_model(model, checkpoint)
    model = model.to(device)
    model.eval()

    dataset = Dataset(opt, monitor_args.monitor_split)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory and use_cuda,
        drop_last=False)

    statistics = GateStatistics()
    processed_batches = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            if (monitor_args.max_batches > 0
                    and batch_index >= monitor_args.max_batches):
                break
            inputs = [frame.to(device=device, non_blocking=True)
                      for frame in batch['input']]
            oracle_gate = batch.get('oracle_gate')
            if oracle_gate is not None:
                oracle_gate = oracle_gate.to(device=device, non_blocking=True)
            [output] = model(inputs, oracle_gate=oracle_gate)
            if not statistics.update_from_output(output):
                raise RuntimeError(
                    'model output has no CPCA gate values; make sure the '
                    'monitoring changes and --use_cpca_gate are enabled')
            processed_batches += 1
            if processed_batches % 20 == 0:
                print('Processed {} / {} batches'.format(
                    processed_batches, len(data_loader)))

    summary = statistics.summary()
    if not summary:
        raise RuntimeError('no samples were processed')
    summary['checkpoint'] = checkpoint
    summary['opt_file'] = os.path.abspath(opt_file)
    summary['dataset'] = opt.dataset
    summary['split'] = monitor_args.monitor_split
    summary['batches'] = processed_batches
    summary['diagnosis'] = _diagnosis(summary)

    output_path = monitor_args.output
    if output_path is None:
        stem, _ = os.path.splitext(checkpoint)
        output_path = '{}.gate_{}.json'.format(
            stem, monitor_args.monitor_split)
    output_path = os.path.abspath(output_path)
    with open(output_path, 'w') as destination:
        json.dump(summary, destination, indent=2, sort_keys=True)

    _print_summary(summary)
    print('Saved JSON report to {}'.format(output_path))


if __name__ == '__main__':
    main()
