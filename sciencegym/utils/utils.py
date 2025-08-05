import json
from pathlib import Path
import numpy as np

def cast_to_serialize(element):
    if isinstance(element, dict):
        return {k: cast_to_serialize(v) for k, v in element.items()}
    if isinstance(element, list):
        return [cast_to_serialize(v) for v in element]
    if isinstance(element, tuple):
        return tuple([cast_to_serialize(v) for v in element])
    if isinstance(element, Path):
        return str(element)
    if isinstance(element, np.ndarray):
        return element.tolist()
    return element


def save_arguments(args):
    print(f"Experiments result will be saved to {args.result_dir}")
    args.result_dir.mkdir(exist_ok=True, parents=True)
    with open(args.result_dir / 'arguments.json', "w") as outfile:
        json.dump(cast_to_serialize(vars(args)), outfile, indent=4)


def save_results(args, results):
    print(f"Results saved in : {args.result_dir / 'results_sr.json'}")
    with open(args.result_dir / 'results_sr.json', "w") as outfile:
        json.dump(cast_to_serialize(results), outfile, indent=4)


def get_exsisting_csv_path(args):
    csv_file = args.root_dir / args.path_to_regression_table
    if not csv_file.exists():
        raise FileExistsError(f"File {csv_file} does not exist")
    return csv_file
