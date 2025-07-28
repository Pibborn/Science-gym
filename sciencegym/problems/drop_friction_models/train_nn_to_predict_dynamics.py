import json
from argparse import ArgumentParser
from pathlib import Path

import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from definitions import ROOT_DIR
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error

from sciencegym.problems.drop_friction_models.preprocess_data import prepare_dataset


def run():
    parser = ArgumentParser(description="Parser for loading data")
    args = parser.parse_args()
    args.system = 'Teflon-Au-Ethylene Glycol'
    args.path_to_datasets = '/home/jbrugger/PycharmProjects/RobotScientist/data/Xiaomei/single_datasets'
    args.save_path = ROOT_DIR / f'environments/drop_friction_models/{args.system}'
    print(f'Model weights,scaler and data are saved and load from: {args.save_path}')

    # training parameters
    args.train_model = True
    args.n_epochs = 300  # number of epochs to run
    args.batch_size = 64  # size of each batch
    args.lr = 0.0001
    args.target = 'friction_force'
    args.corridor_with = 5
    args.delete_adjacent_rows_number = 2

    data = define_data_structure()

    get_files(args, data)
    filtered_dfs = prepare_dataset(args, data['files'])
    get_train_val_test_index(args, data, filtered_dfs)
    fill_data_dict_with_raw_values(data, filtered_dfs)
    scalar_dict = fit_scaler(data)
    fill_data_with_transformed_values(data, scalar_dict)
    fill_data_with_tensor(data)

    if args.train_model:
        model = define_model(
            len(data['in_features']),
            len(data['out_features'])
        )
        metric, best_weights = train_loop(args, model, data)
        save_model_scaler_index_args(args, best_weights, data, model, scalar_dict)
        plot_learning_curve(args, metric)

    _test_model(args, data)


def plot_learning_curve(args, metric):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(metric['history'], label='train')
    fig.suptitle('Model loss')
    fig.tight_layout()
    fig.savefig(args.save_path / 'learning_curve.png')
    fig.show()


def get_files(args, data):
    if (args.save_path / 'data_index.pkl').exists():
        load_data = joblib.load(args.save_path / 'data_index.pkl')
        data['files'] = load_data['files']
    else:
        system_dict = files_to_system_dict(args)
        data['files'] = system_dict[args.system]


def define_data_structure():
    data = {
        'files': [],
        'in_features': ['time', 'tilt_angle'],
        'out_features': ['drop_length',
                         'adv',
                         'rec',
                         'avg_vel',
                         'width',
                         'y'],
        'train': {
            'index': None,
            'raw': {'X': None, 'Y': None},
            'transformed': {'X': None, 'Y': None},
            'tensor': {'X': None, 'Y': None}
        },
        'val': {
            'index': None,
            'raw': {'X': None, 'Y': None},
            'transformed': {'X': None, 'Y': None},
            'tensor': {'X': None, 'Y': None}
        },
        'test': {
            'index': None,
            'raw': {'X': None, 'Y': None},
            'transformed': {'X': None, 'Y': None},
            'tensor': {'X': None, 'Y': None}
        }
    }
    return data


def save_model_scaler_index_args(args, best_weights, data, model, scalar_dict):
    args.save_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(args, args.save_path / 'args.pkl')
    # with open(args.save_path / 'args.json', "w") as outfile:
    #     json.dump(vars(args), outfile, indent=4)
    torch.save(best_weights, args.save_path / 'model_weights.pth')
    joblib.dump(scalar_dict['X'], args.save_path / 'scaler_X.pkl')
    joblib.dump(scalar_dict['Y'], args.save_path / 'scaler_Y.pkl')
    joblib.dump(
        {
            'files': data['files'],
            'train': data['train']['index'],
            'val': data['val']['index'],
            'test': data['test']['index'],
        },
        args.save_path / 'data_index.pkl')


def fit_scaler(data):
    scalar_dict = {
        'X': StandardScaler(),
        'Y': StandardScaler()
    }
    scalar_dict['X'].fit(data['train']['raw']['X'])
    scalar_dict['Y'].fit(data['train']['raw']['Y'])
    return scalar_dict


def fill_data_with_tensor(data):
    data['train']['tensor']['X'] = torch.tensor(data['train']['transformed']['X'], dtype=torch.float32)
    data['train']['tensor']['Y'] = torch.tensor(data['train']['transformed']['Y'], dtype=torch.float32)
    data['val']['tensor']['X'] = torch.tensor(data['val']['transformed']['X'], dtype=torch.float32)
    data['val']['tensor']['Y'] = torch.tensor(data['val']['transformed']['Y'], dtype=torch.float32)
    data['test']['tensor']['X'] = torch.tensor(data['test']['transformed']['X'], dtype=torch.float32)
    data['test']['tensor']['Y'] = torch.tensor(data['test']['transformed']['Y'], dtype=torch.float32)


def fill_data_with_transformed_values(data, scalar_dict):
    data['train']['transformed']['X'] = scalar_dict['X'].transform(data['train']['raw']['X'])
    data['train']['transformed']['Y'] = scalar_dict['Y'].transform(data['train']['raw']['Y'])
    data['val']['transformed']['X'] = scalar_dict['X'].transform(data['val']['raw']['X'])
    data['val']['transformed']['Y'] = scalar_dict['Y'].transform(data['val']['raw']['Y'])
    data['test']['transformed']['X'] = scalar_dict['X'].transform(data['test']['raw']['X'])
    data['test']['transformed']['Y'] = scalar_dict['Y'].transform(data['test']['raw']['Y'])


def fill_data_dict_with_raw_values(data, filtered_dfs):
    data['train']['raw']['X'] = filtered_dfs.iloc[data['train']['index']].loc[:, data['in_features']]
    data['train']['raw']['Y'] = filtered_dfs.iloc[data['train']['index']].loc[:, data['out_features']]
    data['val']['raw']['X'] = filtered_dfs.iloc[data['val']['index']].loc[:, data['in_features']]
    data['val']['raw']['Y'] = filtered_dfs.iloc[data['val']['index']].loc[:, data['out_features']]
    data['test']['raw']['X'] = filtered_dfs.iloc[data['test']['index']].loc[:, data['in_features']]
    data['test']['raw']['Y'] = filtered_dfs.iloc[data['test']['index']].loc[:, data['out_features']]


def get_train_val_test_index(args, data, filtered_dfs):
    if (args.save_path / 'data_index.pkl').exists():
        load_data = joblib.load(args.save_path / 'data_index.pkl')
        data['train']['index'] = load_data['train']
        data['val']['index'] = load_data['val']
        data['test']['index'] = load_data['test']
    else:
        index = np.arange(filtered_dfs.shape[0])
        train_val_split = int(0.6 * (filtered_dfs.shape[0]))
        val_test_split = int(0.8 * (filtered_dfs.shape[0]))
        np.random.shuffle(index)
        data['train']['index'] = index[:train_val_split]
        data['val']['index'] = index[train_val_split:val_test_split]
        data['test']['index'] = index[val_test_split:]


def _test_model(args, data):
    loaded_scaler_Y = joblib.load(ROOT_DIR / f'environments/drop_friction_models/{args.system}/scaler_Y.pkl')
    weights = torch.load(ROOT_DIR / f'environments/drop_friction_models//{args.system}/model_weights.pth')
    model = define_model(len(data['in_features']), len(data['out_features']))
    model.load_state_dict(weights)
    model.eval()

    prediction_dict = {
        'train': {'transformed': None, 'raw': pd.DataFrame()},
        'val': {'transformed': None, 'raw': pd.DataFrame()},
        'test': {'transformed': None, 'raw': pd.DataFrame()}
    }
    for split in ['train', 'val', 'test']:
        prediction_dict[split]['transformed'] = model(
            data[split]['tensor']['X']).detach()

        prediction_dict[split]['raw'] = pd.DataFrame(
            loaded_scaler_Y.inverse_transform(
                prediction_dict[split]['transformed']),
            columns=data[split]['raw']['Y'].columns
        )

    metric = {}
    for split in ['train', 'val', 'test']:
        # Calculate the Root Mean Squared Error (RMSE)
        metric[split] = {}
        metric[split]['rmse'] = np.sqrt(
            mean_squared_error(
                data[split]['raw']['Y'],
                prediction_dict[split]['raw'],
                multioutput='raw_values'
            )
        )
        # Calculate the standard deviation of the true values
        std_dev = np.std(data[split]['raw']['Y'], axis=0)
        # Calculate the Normalized Root Mean Squared Error (NRMSE)
        metric[split]['nrmse'] = metric[split]['rmse'] / std_dev

    fig, axs = plt.subplots(figsize=(14, 25),
                            nrows=len(data['train']['raw']['Y'].columns),
                            sharex=True
                            )
    for i, feature in enumerate(data['train']['raw']['Y'].columns):
        ax = axs[i]
        bias = 0
        title = f'NRMSE {feature}'
        for split in ['train', 'val', 'test']:
            sorted_index = np.argsort(data[split]['raw']['Y'].index.values)
            y_pred = prediction_dict[split]['raw'].loc[:, feature].to_numpy()[sorted_index]
            y_true = data[split]['raw']['Y'].loc[:, feature].to_numpy()[sorted_index]
            index_plot = np.arange(bias, bias + len(y_pred), 1)
            ax.scatter(index_plot, y_true, label=f'true_{split}', s=1)
            ax.scatter(index_plot, y_pred, label=f'pred_{split}', s=1)
            bias += len(y_pred)
            title += f" {split}: {round(metric[split]['nrmse'].loc[feature], 2)}"
        ax.set_title(title)

        ax.legend(loc='upper right')
        ax.set_ylabel('Friction Force' if feature == 'y' else feature)
    ax.set_xlabel('Index in concatenated dataset')
    fig.tight_layout()
    save_path = args.save_path / f'nn_pred_{args.system}.pdf'
    print(f"Saving prediction plot to: {save_path}")
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path)
    plt.show()
    pass


def train_loop(args, model, data):
    metric = {
        'best_val_mse': np.inf,
        'history': []
    }
    best_weights = None

    # loss function and optimizer
    loss_fn = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in tqdm.tqdm(range(args.n_epochs)):
        model.train()
        batch_start = torch.arange(0, data['train']['tensor']['X'].shape[0], args.batch_size)
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = data['train']['tensor']['X'][start:start + args.batch_size]
                y_batch = data['train']['tensor']['Y'][start:start + args.batch_size]
                # forward pass
                y_pred = model(X_batch)
                train_loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                train_loss.backward()
                # update weights
                optimizer.step()
                # print progress
                train_loss = float(train_loss)
                bar.set_postfix(mse=train_loss)
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(data['val']['tensor']['X']).detach()
        val_mse = loss_fn(y_pred, data['val']['tensor']['Y'])
        val_mse = float(val_mse)
        metric['history'].append(val_mse)
        if val_mse < metric['best_val_mse']:
            metric['best_val_mse'] = val_mse
            best_weights = copy.deepcopy(model.state_dict())

    print(f"Val MSE: {metric['best_val_mse']:.3e}")
    print(f"Val RMSE: {np.sqrt(metric['best_val_mse']):.3e}")
    with open(f'{ROOT_DIR}/sciencegym/problems/drop_friction_models/log_learning_model.txt', 'a') as file:
        file.write(f"{args.system:<50}:  Last train MSE: {metric['history'][-1]:.3e}"
                   f" \t Best Val MSE: {metric['best_val_mse']:.3e} \n ")
    model.load_state_dict(best_weights)
    return metric, best_weights


def files_to_system_dict(args):
    files = [f for f in (ROOT_DIR / args.path_to_datasets).iterdir()
             if f.is_file()
             ]
    system_dict = {}
    for file in files:
        name = file.name
        system = ''.join(name.split('_')[1:-1])
        print(system)
        if system not in system_dict:
            system_dict[system] = []
        else:
            system_dict[system].append(file)
    return system_dict


def define_model(len_input, len_output):
    model = nn.Sequential(
        nn.Linear(len_input, 24),
        nn.ReLU(),
        nn.Linear(24, 12),
        nn.ReLU(),
        nn.Linear(12, 12),
        nn.ReLU(),
        nn.Linear(12, len_output),
    )
    return model


if __name__ == '__main__':
    run()
