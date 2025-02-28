import os
import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader


def _time_series_stats(data, mask):
    channels, eps = data.shape[-1], 1e-7
    data = data.transpose((2, 0, 1)).reshape(channels, -1)
    mask = mask.transpose((2, 0, 1)).reshape(channels, -1)
    mu, std = np.zeros((channels, 1)), np.zeros((channels, 1))
    for c in range(channels):
        val = data[c, :][mask[c, :] == 1]
        mu[c], std[c] = np.mean(val), np.std(val)
        std[c] = np.max([std[c][0], eps])
    return mu, std


def _time_normalize_raindrop(data, mask, time, mu, std, t_max):
    num, t, channels = data.shape
    data = data.transpose((2, 0, 1)).reshape(channels, -1)
    mask = mask.transpose((2, 0, 1)).reshape(channels, -1)
    for c in range(channels):
        data[c] = (data[c] - mu[c]) / (std[c] + 1e-18)
    data = data * mask
    data = data.reshape(channels, num, t).transpose((1, 2, 0))
    mask = mask.reshape(channels, num, t).transpose((1, 2, 0))
    time /= t_max
    return np.concatenate([data, mask, time], axis=-1)


def _time_normalize(time, mask):
    time_mask = mask.sum(-1) > 0
    t_max = np.max(time[time_mask])
    time[time_mask] = time[time_mask] / t_max
    return time


def _val_normalize(val, mask):
    val[mask == 0.] = -np.inf
    data_max = np.max(np.max(val, axis=0), axis=0)

    val[mask == 0.] = np.inf
    data_min = np.min(np.min(val, axis=0), axis=0)

    val[mask == 0.], data_max[data_max == 0.] = 0., 1.
    if (data_max != 0.).all():
        val_norm = (val - data_min) / data_max
    else:
        raise Exception('Zeros!')

    if np.isnan(val_norm).all():
        raise Exception('nans!')

    val_norm[mask == 0.] = 0.
    return val_norm


def _to_forecast_format(data: np.ndarray, observed_length: int = -1, forcast_length: int = 12):
    dims = data.shape[-1] // 2
    val, mask, time = data[..., :dims], data[..., dims:-1], data[..., -1:]

    source, target = [], []


    return source, target


def _to_missing(data, ratio, mode, path):
    N, T, dims = data.shape[0], data.shape[1], data.shape[-1] // 2
    num_missing_features = round(ratio * dims)
    if mode == 'sample':
        for i in range(N):
            idx = np.random.choice(dims, num_missing_features, replace=False)
            data[i, :, idx] = np.zeros((T, num_missing_features)).T
    elif mode == 'set':
        score_indices = np.load(path, allow_pickle=True)[:, 0]
        idx = score_indices[:num_missing_features].astype(int)
        data[..., idx] = np.zeros((N, T, num_missing_features))
    return data


def load_data_classification(name, path, args):
    match name.lower():
        case 'phy12':
            filename, missing_file, split_file = 'physionet12.npy', 'saved/IG_density_scores_P12.npy', 'saved/phy12_split' + str(args.n_split) + '.npy'
        case 'phy19':
            filename, missing_file, split_file = 'physionet19.npy', 'saved/IG_density_scores_P19.npy', 'saved/phy19_split' + str(args.n_split) + '_new.npy'
        case 'pam':
            filename, missing_file, split_file = 'pam.npy', 'saved/IG_density_scores_PAM.npy', 'saved/PAM_split_' + str(args.n_split) + '.npy'
        case _:
            filename, missing_file, split_file = name + '.npy', 'saved/IG_density_scores_' + name + '.npy', None

    data = np.load(os.path.join(path, filename), allow_pickle=True).item()
    channels = data['x'].shape[-1] // 2
    val, mask, time, label = data['x'][..., :channels], data['x'][..., channels:-1], data['x'][..., -1:], data['y'].reshape(-1)
    data = np.concatenate([val, mask, time], axis=-1)

    # normalize
    # val, time = _val_normalize(val, mask), _time_normalize(time, mask)

    if split_file is None:
        # random index, training valid and testing number
        assert len(args.split) == 3 and np.sum(args.split) == 1.
        total, ratio = val.shape[0], np.cumsum(args.split)
        perm, index = np.random.permutation(total), (total * ratio).astype(np.int64)
        train_index, valid_index, test_index = perm[0:index[0]], perm[index[0]:index[1]], perm[index[1]:index[2]]
    else:
        train_index, valid_index, test_index = np.load(os.path.join(path, split_file), allow_pickle=True)

    train_x, train_y = data[train_index], label[train_index]
    valid_x, valid_y = data[valid_index], label[valid_index]
    test_x, test_y = data[test_index], label[test_index]

    # normalize
    mu, std = _time_series_stats(train_x[..., :channels], train_x[..., channels:-1])
    t_max = np.max(train_x[..., -1])
    train_x = _time_normalize_raindrop(train_x[..., :channels], train_x[..., channels:-1], train_x[..., -1:], mu, std, t_max)
    valid_x = _time_normalize_raindrop(valid_x[..., :channels], valid_x[..., channels:-1], valid_x[..., -1:], mu, std, t_max)
    test_x = _time_normalize_raindrop(test_x[..., :channels], test_x[..., channels:-1], test_x[..., -1:], mu, std, t_max)

    def to_tensor_combined(x, y, is_int=True):
        y = torch.tensor(y)
        return TensorDataset(torch.tensor(x), y.long() if is_int else y)

    n_classes = 0
    match args.downstream.lower():
        case 'classification':
            if args.missing > 0:
                valid_x = _to_missing(valid_x, args.missing, args.missing_mode, os.path.join(path, missing_file))
                test_x = _to_missing(test_x, args.missing, args.missing_mode, os.path.join(path, missing_file))

            if name.lower() in ['pam']:
                metric = ['acc', 'precision', 'recall', 'f1']
            else:
                metric = ['auroc', 'auprc', 'acc']

            n_classes = np.unique(label).size
            train_data = to_tensor_combined(train_x, train_y)
            valid_data = to_tensor_combined(valid_x, valid_y)
            test_data = to_tensor_combined(test_x, test_y)
        case 'forecast':
            raise NotImplementedError
        case 'anomaly':
            raise NotImplementedError
        case _:
            raise NotImplementedError

    dloader_config = {
        'batch_size': args.batch_size,
        'drop_last': True,
        'pin_memory': True,
        'num_workers': 4
    }

    train_dataloader = DataLoader(train_data, shuffle=True, **dloader_config)
    valid_dataloader = DataLoader(valid_data, shuffle=False, **dloader_config)
    test_dataloader = DataLoader(test_data, shuffle=False, **dloader_config)

    return {
        'dims': channels,
        'train': train_dataloader,
        'valid': valid_dataloader,
        'test': test_dataloader,
        'metric': metric,
        'n_class': n_classes,
        'observation_time': 1,
    }


def load_data_forecast(name, path, args):
    match name.lower():
        case 'ushcn':
            from tsdm.tasks import USHCN_DeBrouwer2019
            dataset = USHCN_DeBrouwer2019(normalize_time=True, condition_time=args.cond_time, forecast_horizon=args.forc_time, num_folds=args.nfolds)
        case 'mimiciii':
            from tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019
            dataset = MIMIC_III_DeBrouwer2019(normalize_time=True, condition_time=args.cond_time, forecast_horizon=args.forc_time, num_folds=args.nfolds)
        case 'mimiciv':
            from tsdm.tasks.mimic_iv_bilos2021 import MIMIC_IV_Bilos2021
            dataset = MIMIC_IV_Bilos2021(normalize_time=True, condition_time=args.cond_time, forecast_horizon=args.forc_time, num_folds=args.nfolds)
        case 'physionet2012':
            from tsdm.tasks.physionet2012 import Physionet2012
            dataset = Physionet2012(normalize_time=True, condition_time=args.cond_time, forecast_horizon=args.forc_time, num_folds=args.nfolds)

    from tsdm.collate import tsdm_collate

    dloader_config_train = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": True,
        "pin_memory": True,
        "num_workers": 4,
        "collate_fn": tsdm_collate,
    }

    dloader_config_infer = {
        "batch_size": 64,
        "shuffle": False,
        "drop_last": False,
        "pin_memory": True,
        "num_workers": 4,
        "collate_fn": tsdm_collate,
    }

    train_loader = dataset.get_dataloader((args.fold, 'train'), **dloader_config_train)
    valid_loader = dataset.get_dataloader((args.fold, 'valid'), **dloader_config_infer)
    test_loader = dataset.get_dataloader((args.fold, 'test'), **dloader_config_infer)

    return {
        'dims': dataset.dataset.shape[-1],
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader,
        'metric': ['mse'],
        'n_class': None,
        'observation_time': dataset.observation_time,
    }


def load_physionet_data(args, device, q, flag):
    from sklearn import model_selection
    from physionet import PhysioNet, get_data_min_max, variable_time_collate_fn2

    def normalize_masked_data(data, mask, att_min, att_max):
        # we don't want to divide by zero
        att_max[att_max == 0.] = 1.

        if (att_max != 0.).all():
            data_norm = (data - att_min) / att_max
        else:
            raise Exception("Zero!")

        if torch.isnan(data_norm).any():
            raise Exception("nans!")

        # set masked out elements back to zero
        data_norm[mask == 0] = 0

        return data_norm, att_min, att_max

    def variable_time_collate_fn(batch, device=torch.device("cpu"), classify=False, activity=False,
                                 data_min=None, data_max=None):
        """
        Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
          - record_id is a patient id
          - tt is a 1-dimensional tensor containing T time values of observations.
          - vals is a (T, D) tensor containing observed values for D variables.
          - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
          - labels is a list of labels for the current patient, if labels are available. Otherwise None.
        Returns:
          combined_tt: The union of all time observations.
          combined_vals: (M, T, D) tensor containing the observed values.
          combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
        """
        D = batch[0][2].shape[1]
        # number of labels
        N = batch[0][-1].shape[1] if activity else 1
        len_tt = [ex[1].size(0) for ex in batch]
        maxlen = np.max(len_tt)
        enc_combined_tt = torch.zeros([len(batch), maxlen]).to(device)
        enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
        enc_combined_mask = torch.zeros([len(batch), maxlen, D]).to(device)
        if classify:
            if activity:
                combined_labels = torch.zeros([len(batch), maxlen, N]).to(device)
            else:
                combined_labels = torch.zeros([len(batch), N]).to(device)

        for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
            currlen = tt.size(0)
            enc_combined_tt[b, :currlen] = tt.to(device)
            enc_combined_vals[b, :currlen] = vals.to(device)
            enc_combined_mask[b, :currlen] = mask.to(device)
            if classify:
                if activity:
                    combined_labels[b, :currlen] = labels.to(device)
                else:
                    combined_labels[b] = labels.to(device)

        if not activity:
            enc_combined_vals, _, _ = normalize_masked_data(enc_combined_vals, enc_combined_mask,
                                                            att_min=data_min, att_max=data_max)

        if torch.max(enc_combined_tt) != 0.:
            enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)

        combined_data = torch.cat(
            (enc_combined_vals, enc_combined_mask, enc_combined_tt.unsqueeze(-1)), 2)
        if classify:
            return combined_data, combined_labels
        else:
            return combined_data

    train_dataset_obj = PhysioNet('data/physionet', train=True,
                                  quantization=q,
                                  download=True, n_samples=min(10000, args.n),
                                  device=device)
    # Use custom collate_fn to combine samples with arbitrary time observations.
    # Returns the dataset along with mask and time steps
    test_dataset_obj = PhysioNet('data/physionet', train=False,
                                 quantization=q,
                                 download=True, n_samples=min(10000, args.n),
                                 device=device)

    # Combine and shuffle samples from physionet Train and physionet Test
    total_dataset = train_dataset_obj[:len(train_dataset_obj)]

    if not args.classif:
        # Concatenate samples from original Train and Test sets
        # Only 'training' physionet samples are have labels.
        # Therefore, if we do classifiction task, we don't need physionet 'test' samples.
        total_dataset = total_dataset + test_dataset_obj[:len(test_dataset_obj)]
    # print(len(total_dataset))
    # Shuffle and split
    train_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8, random_state=42, shuffle=True)

    record_id, tt, vals, mask, labels = train_data[0]

    # n_samples = len(total_dataset)
    input_dim = vals.size(-1)
    data_min, data_max = get_data_min_max(total_dataset, device)
    batch_size = min(min(len(train_dataset_obj), args.batch_size), args.n)
    if flag:
        test_data_combined = variable_time_collate_fn(test_data, device, classify=args.classif,
                                                      data_min=data_min, data_max=data_max)

        if args.classif:
            train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8,
                                                                    random_state=11, shuffle=True)
            train_data_combined = variable_time_collate_fn(
                train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
            val_data_combined = variable_time_collate_fn(
                val_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
            print(train_data_combined[1].sum(
            ), val_data_combined[1].sum(), test_data_combined[1].sum())
            print(train_data_combined[0].size(), train_data_combined[1].size(),
                  val_data_combined[0].size(), val_data_combined[1].size(),
                  test_data_combined[0].size(), test_data_combined[1].size())

            timestamp_dim = train_data_combined[0].size(1)

            train_data_combined = TensorDataset(
                train_data_combined[0], train_data_combined[1].long().squeeze())
            val_data_combined = TensorDataset(
                val_data_combined[0], val_data_combined[1].long().squeeze())
            test_data_combined = TensorDataset(
                test_data_combined[0], test_data_combined[1].long().squeeze())
        else:
            train_data_combined = variable_time_collate_fn(
                train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
            # print(train_data_combined.size(), test_data_combined.size())
            pred_len = np.max([train_data_combined.size(1), test_data_combined.size(1)])

            timestamp_dim = train_data_combined.size(1)

        train_dataloader = DataLoader(
            train_data_combined, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(
            test_data_combined, batch_size=batch_size, shuffle=False)

    else:
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False,
                                      collate_fn=lambda batch: variable_time_collate_fn2(batch, args, device, data_type="train",
                                                                                         data_min=data_min, data_max=data_max))
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                     collate_fn=lambda batch: variable_time_collate_fn2(batch, args, device, data_type="test",
                                                                                        data_min=data_min, data_max=data_max))

    return {
        'dims': input_dim,
        'train': train_dataloader,
        'valid': test_dataloader,              # for simple and the same as mtans
        'test': test_dataloader,
        'metric': ['mse'],
        'n_class': None,
        'observation_time': timestamp_dim,
        'pred_len': pred_len,
    }


def load_data(name, path, args):
    if args.downstream.lower() in ['classification']:
        obj = load_data_classification(name, path, args)
    elif args.downstream.lower() in ['forecast']:
        obj = load_data_forecast(name, path, args)
    elif args.downstream.lower() in ['impute']:
        import argparse
        params = argparse.Namespace()
        params.classif = False
        params.n = 8000
        params.batch_size = args.batch_size
        obj = load_physionet_data(params, 'cpu', 0.016, flag=1)
    else:
        raise NotImplementedError('Not implemented %s downstream!' % args.downstream)

    return obj

