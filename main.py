import argparse
import random
import sys
import os
import time
import numpy as np

import torch
from torch.optim import AdamW

from evaluate import evaluate, mse_with_mask_torch, cross_entropy
from datasets import load_data
from models.model import BackBone
from utils import subsample_timepoints


def parse_args():
    parser = argparse.ArgumentParser(description='Implementation of LGTS.')

    parser.add_argument('-ds', '--downstream', default='classification', type=str,
                        choices=['classification', 'forecast', 'impute', 'anomaly', 'self-supervised'])
    parser.add_argument('-da', '--dataset', default=None, type=str, required=True)
    parser.add_argument('-p', '--path', default='./data', type=str)
    parser.add_argument('-sp', '--split', default=[0.8, 0.1, 0.1], nargs='+', type=float)
    parser.add_argument('-ns', '--n-split', default=1, type=int)

    parser.add_argument('-s', '--seed', default=0, type=int)
    parser.add_argument('-e', '--epochs', default=200, type=int)
    parser.add_argument('-bs', '--batch-size', default=32, type=int)
    parser.add_argument('-lr', '--learning-rate', default=1e-3, type=float)

    # model
    parser.add_argument('-rf', '--reference-points', default=128, type=int)
    parser.add_argument('-pa', '--patch', default=8, type=int)
    parser.add_argument('-pm', '--patch-mode', default='manner', type=str, choices=['manner', 'auto'])

    # # graph
    parser.add_argument('-l', '--layers', default=2, type=int)
    parser.add_argument('-ld', '--latent-dims', default=128, type=int)
    parser.add_argument('-ah', '--attention-head', default=1, type=int)

    # # transformer
    parser.add_argument('-tf', '--transformer-factor', default=1, type=int)
    parser.add_argument('-tl', '--transformer-layers', default=3, type=int)
    parser.add_argument('-th', '--transformer-heads', default=8, type=int)
    parser.add_argument('-td', '--transformer-dims', default=256, type=int)
    parser.add_argument('-ta', '--transformer-activation', default='gelu', type=str)
    parser.add_argument('-toa', '--transformer-attention', action=argparse.BooleanOptionalAction)

    parser.add_argument('-dr', '--dropout', default=0.1, type=float)

    # forecasting
    parser.add_argument('-ct', '--cond_time', type=int, default=36)
    parser.add_argument('-ft', '--forc_time', type=int, default=0)
    parser.add_argument('-nf', '--nfolds', type=int, default=5)
    parser.add_argument('-fi', '--fold', type=int, default=0)

    # classification missing
    parser.add_argument('-m', '--missing', default=0, type=float)
    parser.add_argument('-mm', '--missing-mode', default='sample', type=str, choices=['sample', 'set'])

    # imputation
    parser.add_argument('-stp', '--sample-tp', default=1, type=float)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # fixed seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load data
    object = load_data(args.dataset, args.path, args)

    train_loader, valid_loader, test_loader = object['train'], object['valid'], object['test']
    channels, metric = object['dims'], object['metric']

    config, criterion = {'obs': object['observation_time']}, None
    if args.downstream.lower() in ['classification']:
        config['n_class'] = object['n_class']
        criterion = torch.nn.CrossEntropyLoss()
        # criterion = cross_entropy
    elif args.downstream.lower() in ['forecast']:
        config['pred_len'] = 3 if args.forc_time == 0 else args.forc_time
        criterion = mse_with_mask_torch
    elif args.downstream.lower() in ['impute']:
        config['pred_len'] = object['pred_len']
        criterion = mse_with_mask_torch
    else:
        pass

    match args.patch_mode.lower():
        case _:
            n_patches = args.patch

    model = BackBone(channels=channels, attn_head=args.attention_head, latent_dim=args.latent_dims, n_layers=args.layers,
                     ref_points=args.reference_points, n_patches=n_patches, dropout=args.dropout, former_factor=args.transformer_factor,
                     former_dff=args.transformer_dims, former_output_attention=args.transformer_attention, former_layers=args.transformer_layers,
                     former_heads=args.transformer_heads, former_activation=args.transformer_activation, downstream=args.downstream, config=config)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, min_lr=1e-5, verbose=True)

    def print_info(info: dict, phase, epoch, max_epoch):
        sys.stdout.write('%s Phase, Epoch#%.2d/%d' % (phase, epoch, max_epoch))
        for k, v in info.items():
            sys.stdout.write(', %s = %.5f' % (k, v))
        sys.stdout.write('\n')

    for epoch in range(args.epochs):
        loss_list, time_list, sch_loss = [], [], torch.tensor(0., device=device)
        for ind, batch in enumerate(train_loader):
            start_time = time.time()
            if args.downstream.lower() in ['impute']:
                batch = batch.to(device)
                if args.sample_tp < 1:
                    val, mask, tp = batch[..., :channels], batch[..., channels:-1], batch[..., -1]
                    sub_data, sub_tp, sub_mask = subsample_timepoints(val.clone(), tp.clone(), mask.clone(), args.sample_tp)
                    train = torch.cat([sub_data, sub_mask, sub_tp.unsqueeze(-1)], dim=-1)
            else:
                train, target = batch
                train, target = train.to(device), target.to(device)

            # pred, recon_loss = model(train, target[..., -1]), torch.tensor(0., device=device)
            pred, recon_loss = model(train, None), torch.tensor(0., device=device)

            if args.downstream == 'forecast' and pred.size() != target[..., :channels].size():
                print('ok')

            if args.downstream.lower() in ['forecast']:
                loss = criterion(pred, target[..., :channels], target[..., channels:-1])
            elif args.downstream.lower() in ['impute']:
                loss = criterion(pred[:, :batch.size(1), :], batch[..., :channels], batch[..., channels:-1])
            else:
                loss = criterion(pred, target)

            loss += recon_loss
            sch_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item()), time_list.append(time.time() - start_time)

            if (ind + 1) % 20 == 0:
                print('Epoch#%.2d/%d, Iter#%.2d/%d, time = %.4fs, loss = %.4f, recon = %.4f' %
                      (epoch + 1, args.epochs, ind + 1, len(train_loader), np.mean(time_list), np.mean(loss_list), recon_loss.item()))
            # end if ind
        # end for ind
        scheduler.step(sch_loss / len(train_loader))

        valid_info = evaluate(model, valid_loader, metric, device, args)
        print_info(valid_info, 'Valid', epoch + 1, args.epochs)
        eval_info = evaluate(model, test_loader, metric, device, args) if args.downstream.lower() not in ['impute'] else valid_info
        print_info(eval_info, 'Test', epoch + 1, args.epochs)
        sys.stdout.write('\n')
