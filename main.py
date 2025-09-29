#!/usr/bin/env python

import sys
import argparse
from src.experiments.run import run_experiments

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Запуск экспериментов')
    parser.add_argument(
        '--dataset', choices=['raw', 'clean'], default='clean',
        help='Выбор набора данных: raw или clean'
    )
    args = parser.parse_args()

    run_experiments('config_only_med.yaml', args.dataset)
