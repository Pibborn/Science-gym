import random
import pandas as pd
import numpy as np


def prepare_dataset(args, files):
    filtered_dfs = []
    for f in files:
        df = load_xiaomei_single_dataset(args, f)
        filtered_df = filter(df, args)
        filtered_dfs.append(filtered_df)
    filtered_dfs = pd.concat(filtered_dfs, axis=0, ignore_index=True)
    unique_values = filtered_dfs.loc[:, 'excel_name'].unique()
    if len(unique_values) > 1:
        print(f"Found {len(unique_values)} unique values")
        print(f'Unique values: {unique_values}')
    print(f'We have {len(files)} files with {len(filtered_dfs)} items')
    return filtered_dfs


def filter(df, args):
    median = df['y'].quantile(0.5)

    # Step 2: Calculate the interquartile range (IQR)
    q75 = df['y'].quantile(0.75)
    q25 = df['y'].quantile(0.25)
    iqr = q75 - q25
    # Step 3: Define the corridor limits
    lower_bound = median - args.corridor_with * iqr
    upper_bound = median + args.corridor_with * iqr

    # Step 4: Find indices of rows outside the corridor
    out_of_corridor = df[(df['y'] < lower_bound) | (df['y'] > upper_bound)].index

    # Step 5: Create a set of indices to delete (including two rows before and two after)
    rows_to_delete = set(out_of_corridor)
    for idx in out_of_corridor:
        index_before = idx - args.delete_adjacent_rows_number
        if index_before < df.index[0]:
            index_before = df.index[0]
        index_after = idx + args.delete_adjacent_rows_number
        if index_after > df.index[-1]:
            index_after = df.index[-1]
        # Add x rows before and x rows after the row outside the corridor
        rows_to_delete.update(range(index_before, index_after + 1))
    # Step 6: Remove rows that are either out of the corridor or around them
    if len(rows_to_delete) > 0:
        print(f"Removing {len(rows_to_delete)} rows from {df.iloc[0].loc[['row_id', 'col_id', 'excel_name']].to_dict()}")
    filtered_df = df.drop(rows_to_delete)
    return filtered_df


def get_unit_dict(args):
    df_units = pd.read_csv(args.ROOT_DIR / args.path_to_units)
    units = {}
    for i in range(len(df_units["Variable"])):
        val = [df_units["m"][i], df_units["s"][i], df_units["kg"][i], df_units["T"][i], df_units["V"][i]]
        val = np.array(val)
        units[df_units["Variable"][i]] = val
    return units


def unit_vector_to_str(vector):
    units_str_list = []
    for i, unit in enumerate(['m', 's', 'kg', 'T', 'V']):
        if not vector[i] == 0.:
            units_str_list.append(f"{unit}^{vector[i]}")
    unit_str = ' * '.join(units_str_list)
    return unit_str


def split_train_test(files):
    id_seen = set()
    train_files = []
    test_files = []
    for file in files:
        id = '_'.join(file.name.split('_')[1:])
        if id in id_seen:
            train_files.append(file)
        else:
            test_files.append(file)
            id_seen.add(id)
    return train_files, test_files



def load_xiaomei_single_dataset(args, path):
    df = pd.read_csv(path,index_col=0)
    df.columns = [s.strip() for s in df.columns]
    df['gamma'] = df.loc[:, 'gamma'].to_numpy() * 0.001 # gamma is given as mN in the Dataset
    df['viscosity'] = df.loc[:, 'viscosity'].to_numpy() * 0.001
    df['adv'] = np.deg2rad(df.loc[:, 'adv'].to_numpy())
    df['rec'] = np.deg2rad(df.loc[:, 'rec'].to_numpy())
    df['tilt_angle'] = np.deg2rad(df.loc[:, 'tilt_angle'].to_numpy())
    df.rename(columns={args.target: 'y'}, inplace=True)

    return df
