import torch
import numpy as np
from sklearn import metrics
from utils import subsample_timepoints


def mse_with_mask(x, x_hat, mask):
    return np.power((x - x_hat) * mask, 2).sum() / (mask.sum())


def mse_with_mask_torch(x, x_hat, mask):
    return torch.pow((x - x_hat) * mask, 2).sum() / (mask.sum())


def cross_entropy(p, y):
    ce, loss = torch.nn.functional.cross_entropy(p, y, reduction='none'), torch.tensor(0., device=p.device)
    for k in y.unique():
        loss += ce[y == k].mean()
    return loss / y.unique().size(0)


def evaluate(model, loader, metric, device, args):
    if args.downstream.lower() in ['impute']:
        info = evaluate_impute(model, loader, metric, device, args)
    else:
        info = evaluate_others(model, loader, metric, device, args)
    return info


def evaluate_impute(model, loader, metric, device, args):
    model.eval()
    pred, truth, info = [], [], {}
    with torch.no_grad():
        for batch in loader:
            batch, channels = batch.to(device), batch.size(-1) // 2
            if args.sample_tp < 1:
                val, mask, tp = batch[..., :channels], batch[..., channels:-1], batch[..., -1]
                sub_data, sub_tp, sub_mask = subsample_timepoints(val.clone(), tp.clone(), mask.clone(), args.sample_tp)
                train = torch.cat([sub_data, sub_mask, sub_tp.unsqueeze(-1)], dim=-1)
            pred.append(model(train, None).detach().cpu().numpy())
            truth.append(batch.detach().cpu().numpy())
        pred, truth = np.concatenate(pred, axis=0), np.concatenate(truth, axis=0)
    model.train()
    info['mse'] = mse_with_mask(pred, truth[:, :pred.shape[1], :pred.shape[-1]], truth[:, :pred.shape[1], pred.shape[-1]:-1])
    return info


def evaluate_others(model, loader, metric, device, args):
    model.eval()
    pred, truth, info = [], [], {}
    with torch.no_grad():
        for batch in loader:
            data, target = batch[0].to(device), batch[1].to(device)
            pred.append(model(data, target[..., -1]).detach().cpu().numpy())
            truth.append(target.detach().cpu().numpy())
        pred, truth = np.concatenate(pred, axis=0), np.concatenate(truth, axis=0)

        average = 'macro' if pred.shape[-1] > 2 else 'binary'
        for measure in metric:
            match measure.lower():
                case 'acc':
                    info[measure] = np.mean(pred.argmax(1) == truth)
                case 'auroc':
                    info[measure] = metrics.roc_auc_score(truth, pred[:, 1])
                case 'auprc':
                    info[measure] = metrics.average_precision_score(truth, pred[:, 1])
                case 'f1':
                    info[measure] = metrics.f1_score(truth, pred.argmax(1), average=average, zero_division=1.)
                case 'recall':
                    info[measure] = metrics.recall_score(truth, pred.argmax(1), average=average, zero_division=1.)
                case 'precision':
                    info[measure] = metrics.precision_score(truth, pred.argmax(1), average=average, zero_division=1.)
                case 'mse':
                    info[measure] = mse_with_mask(pred, truth[..., :pred.shape[-1]], truth[..., pred.shape[-1]:-1])
                case _:
                    raise NotImplementedError
    model.train()
    return info