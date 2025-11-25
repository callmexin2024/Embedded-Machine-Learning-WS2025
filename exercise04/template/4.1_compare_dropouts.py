#!/usr/bin/env python3
"""Plot test_accuracy across different dropout parameters.

Usage:
  python compare_dropouts.py --dir ./results --outdir ./plots
  python compare_dropouts.py --csvs file1.csv file2.csv --outdir ./plots
"""
import argparse
import os
import re
import csv
import matplotlib.pyplot as plt


def read_csv(path):
    epochs = []
    test_acc = []
    test_loss = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            test_acc.append(float(row['test_accuracy']))
            test_loss.append(float(row.get('test_loss', 0.0)))
    return epochs, test_acc, test_loss


def parse_dropout_from_name(name):
    m = re.search(r'dropout([0-9]+\.?[0-9]*)', name)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def collect_csvs_from_dir(d):
    files = []
    for fn in os.listdir(d):
        if fn.lower().endswith('.csv') and 'dropout' in fn:
            files.append(os.path.join(d, fn))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dir', help='Directory containing CSVs (will pick files with "dropout" in name)')
    group.add_argument('--csvs', nargs='+', help='List of CSV files to compare')
    parser.add_argument('--outdir', default='plots', help='Output directory for plot')
    args = parser.parse_args()

    if args.dir:
        csvs = collect_csvs_from_dir(args.dir)
    else:
        csvs = args.csvs

    if not csvs:
        print('No CSV files found to compare.')
        return

    # Prepare datasets with parsed dropout
    datasets = []
    for p in csvs:
        epochs, test_acc, test_loss = read_csv(p)
        name = os.path.splitext(os.path.basename(p))[0]
        dropout = parse_dropout_from_name(name)
        datasets.append({'path': p, 'name': name, 'dropout': dropout, 'epochs': epochs, 'test_acc': test_acc, 'test_loss': test_loss})

    # Sort by dropout value when available, otherwise by name
    datasets.sort(key=lambda d: (d['dropout'] is None, d['dropout'] if d['dropout'] is not None else d['name']))

    os.makedirs(args.outdir, exist_ok=True)
    # Plot test_loss comparison
    plt.figure(figsize=(10, 6))
    for d in datasets:
        label = f"dropout={d['dropout']}" if d['dropout'] is not None else d['name']
        plt.plot(d['epochs'], d['test_loss'], marker='o', label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Test Loss across Dropout Values')
    plt.grid(True)
    plt.legend(fontsize='small')
    outpath_loss = os.path.join(args.outdir, 'dropout_test_loss_comparison.png')
    plt.tight_layout()
    plt.savefig(outpath_loss)
    print(f'Saved dropout loss comparison to {outpath_loss}')

    # Plot test_accuracy comparison
    plt.figure(figsize=(10, 6))
    for d in datasets:
        label = f"dropout={d['dropout']}" if d['dropout'] is not None else d['name']
        plt.plot(d['epochs'], d['test_acc'], marker='o', label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Test Accuracy across Dropout Values')
    plt.grid(True)
    plt.legend(fontsize='small')
    outpath_acc = os.path.join(args.outdir, 'dropout_test_accuracy_comparison.png')
    plt.tight_layout()
    plt.savefig(outpath_acc)
    print(f'Saved dropout accuracy comparison to {outpath_acc}')


if __name__ == '__main__':
    main()
