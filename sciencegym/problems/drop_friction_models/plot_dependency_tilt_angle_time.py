from argparse import ArgumentParser

import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from definitions import ROOT_DIR
from sciencegym.problems.drop_friction_models.preprocess_data import prepare_dataset
from sciencegym.problems.drop_friction_models.train_nn_to_predict_dynamics import define_data_structure, get_files


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

    loaded_scaler_X = joblib.load(ROOT_DIR / f'environments/drop_friction_models/{args.system}/scaler_X.pkl')

    data = define_data_structure()
    get_files(args, data)
    filtered_dfs = prepare_dataset(args, data['files'])
    plot_scaled_input_features(filtered_dfs, loaded_scaler_X)

    plot_scaled_output_features(args, filtered_dfs)




def plot_scaled_input_features(filtered_dfs, loaded_scaler_X):
    tilt_angle_array = filtered_dfs.loc[:, 'tilt_angle'].to_numpy()
    time_array = filtered_dfs.loc[:, 'time'].to_numpy()
    X = loaded_scaler_X.transform(np.column_stack((time_array, tilt_angle_array)))
    time_array = X[:, 0]
    tilt_angle_array = X[:, 1]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(tilt_angle_array, time_array, label='true data')
    x = np.arange(-1.7, 1.7, 0.05)
    y = np.exp(-1 * x +0.0)
    ax.plot(x, y, label='border', color='red')

    y = np.arange(-1.2, 6.2, 1)
    ax.plot([-1.7 for i in range(len(y))], y, color='red')

    y = np.arange(-1.2, 0.3, 0.1)
    ax.plot([1.7 for i in range(len(y))], y, color='red')
    ax.set(xlabel='tilt angle', ylabel='time')
    fig.tight_layout()
    # fig.savefig(args.save_path / 'learning_curve.png')
    fig.show()


def plot_scaled_output_features(args, filtered_dfs):
    loaded_scaler_Y = joblib.load(ROOT_DIR / f'environments/drop_friction_models/{args.system}/scaler_Y.pkl')
    Y = filtered_dfs.loc[:, ['drop_length', 'adv', 'rec', 'avg_vel', 'width', 'y']]
    Y = pd.DataFrame(loaded_scaler_Y.transform(Y),
                     columns=['drop_length', 'adv', 'rec', 'avg_vel', 'width', 'y'])
    fig, axs = plt.subplots(figsize=(4, 4), nrows=3, ncols=2)
    for i, feature in enumerate(['drop_length', 'adv', 'rec', 'avg_vel', 'width', 'y']):
        ax = axs[int(i / 2), i % 2]
        x = Y.loc[:, feature]
        ax.scatter(x, [1 for i in range(len(x))])
        print(f"{feature}: min {round(x.min(), 2)} , max {round(x.max(), 2)}  ")
        ax.set_title(feature)
    fig.tight_layout()
    # fig.savefig(args.save_path / 'learning_curve.png')
    fig.show()


if __name__ == '__main__':
    run()