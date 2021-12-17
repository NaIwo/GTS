from typing import Dict, List
import argparse
from statistics import mean
from copy import deepcopy
import yaml


def add_to_dict(results: Dict, model: str, dataset: str, task_name: str, metric: str, score: float) -> None:
    if model not in results.keys():
        results[model]: Dict = dict()
    if dataset not in results[model].keys():
        results[model][dataset]: Dict = dict()
    if task_name not in results[model][dataset].keys():
        results[model][dataset][task_name]: Dict = dict()
    if metric not in results[model][dataset][task_name].keys():
        results[model][dataset][task_name][metric]: List = list()
    results[model][dataset][task_name][metric].append(score)


def parse(logfile: str) -> Dict:
    datasets: List = ['lap14', 'res14', 'res15', 'res16']
    models: List = ['bert', 'cnn', 'bilstm']
    results: Dict = dict()

    with open(logfile) as f:
        for line in f.readlines():
            line: str = line.strip()
            if 'model type' in line.lower():
                model: str = line[line.rfind(': ') + 1:].strip()
            if 'dataset' in line.lower():
                dataset: str = line[line.rfind(': ') + 1:].strip()
                test_set_started: bool = False
                test_set_line_counter: int = 0
            if 'test set' in line.lower():
                test_set_started: bool = True
            if test_set_started:
                test_set_line_counter += 1
            if test_set_line_counter >= 5:
                idx = line.rfind(': ')
                if idx == -1:
                    task_name: str = line[:-1]
                else:
                    metric: str = line[:idx]
                    score: float = float(line[idx + 1:])
                    add_to_dict(results, model, dataset, task_name, metric, score)
    return results


def get_logfile_name() -> str:
    parser = argparse.ArgumentParser(description='Logfile parser - final results.')
    parser.add_argument('-f', '--logfile', type=str, help='Name of logfile to parse.', required=True)

    args = parser.parse_args()

    logfile: str = args.logfile
    if '.log' not in logfile:
        logfile += '.log'
    return logfile


def get_average_scores(results: Dict) -> None:
    for key, value in results.items():
        if isinstance(value, list):
            results[key] = mean(value)
        if isinstance(value, dict):
            get_average_scores(value)


def save_to_file(results: Dict) -> None:
    with open('results.yml', 'a') as outfile:
        yaml.dump(results, outfile, default_flow_style=False)


if __name__ == '__main__':
    results: Dict = parse(get_logfile_name())
    average_results: Dict = deepcopy(results)
    get_average_scores(average_results)
    save_to_file(average_results)
