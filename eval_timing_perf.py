import argparse
import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import pandas as pd
import pdb
import seaborn as sns
sns.set_style("whitegrid")
import sys

from matplotlib import pyplot as plt


"""Collects and reports timing performance of all models.

  Collects the timing_perf.txt files in each results directory,
  and builds summarizing plots to show data and time efficiency.

"""


DIM = 10

df = pd.DataFrame() 

base_path = '/home/maurice/iwgn_multivariate/results'
model_run_names = [n for n in os.listdir(base_path) if 'dim'+str(DIM) in n]
# For each model in the desired set, add the timing performance data to an array.
for model_run_name in model_run_names:
    run_path = os.path.join(base_path, model_run_name)
    try:
        filename = os.path.join(run_path, 'timing_perf.txt')
        run_performance = np.loadtxt(filename,
            dtype={'names': ('model', 'tag', 'step', 'mmd', 'energy', 'kl', 'time'),
                   'formats': ('|S15', '|S15', np.int, np.float, np.float, np.float, np.float)},
            delimiter=',', skiprows=0)

        for row in run_performance:
            # Do some renaming, for consistency with paper.
            m = row['model']
            t = row['tag']
            if m == 'ce':
                if t.startswith('iw'):
                    row['model'] = 'IW-CE'
                elif t.startswith('miw'):
                    row['model'] = 'MIW-CE'
                elif t.startswith('sn'):
                    row['model'] = 'SNIW-CE'
                elif t.startswith('cond'):
                    row['model'] = 'CGAN-CE'
                elif t.startswith('upsampl'):
                    row['model'] = 'Upsampling-CE'
            elif m == 'mmd':
                if t.startswith('iw'):
                    row['model'] = 'IW-MMD'
                elif t.startswith('miw'):
                    row['model'] = 'MIW-MMD'
                elif t.startswith('sn'):
                    row['model'] = 'SNIW-MMD'
                elif t.startswith('upsampl'):
                    row['model'] = 'Upsampling-MMD'

            # Add row to dataframe.
            row_df = pd.DataFrame(
                [[row['model'], row['tag'], row['step'], row['mmd'],
                 row['energy'], row['kl'], row['time']]], 
                columns=['model', 'tag', 'step', 'mmd', 'energy', 'kl', 'time'])
            df = df.append(row_df, ignore_index=True)

    except:
        print('Did not find timing_perf.txt in {}.'.format(run_path))
        continue

print(df.shape)

####################################
# PLOT JUST CE, JUST MMD, and ALL.
# CE vs Upsampling.
df_ce = df.loc[df['model'].isin(['IW-CE', 'MIW-CE', 'SNIW-CE', 'Upsampling-CE'])]
df_mmd = df.loc[df['model'].isin(['IW-MMD', 'MIW-MMD', 'SNIW-MMD', 'Upsampling-MMD'])]

def boxplots(df_, hue_order_, tag_, measure='mmd'):
    # Plot loss on y-axis and step on x-axis, and brake out by model.
    p = sns.boxplot(
        x='step', y=measure, hue='model', data=df_,
        hue_order=hue_order_,
        palette=sns.color_palette("hls", len(hue_order_)),
        width=1, fliersize=2, linewidth=0.5)
    p.set_xticklabels(p.get_xticklabels(), rotation=30)
    fig = p.get_figure()
    fig.set_size_inches(10,7)
    fig.savefig('plots/boxplot_{}_loss_{}.png'.format(tag_, measure))
    plt.close()
    # Plot time on y-axis and model on x-axis.
    p = sns.boxplot(
        x='model', y='time', data=df_, 
        order=hue_order_,
        palette=sns.color_palette("hls", len(hue_order_)),
        width=1, fliersize=2, linewidth=0.5)
    p.set_xticklabels(p.get_xticklabels(), rotation=30)
    fig = p.get_figure()
    fig.set_size_inches(10,7)
    fig.savefig('plots/boxplot_{}_time.png'.format(tag_))
    plt.close()

_measure = 'mmd'
try:
    boxplots(df_ce, ['IW-CE', 'MIW-CE', 'SNIW-CE', 'Upsampling-CE'], 'ce',
             measure=_measure)
except: print('check that df_ce has items')
try:
    boxplots(df_mmd, ['IW-MMD', 'MIW-MMD', 'SNIW-MMD', 'Upsampling-MMD'], 'mmd',
             measure=_measure)
except: print('check that df_mmd has items')
try:
    boxplots(df,
        ['IW-CE', 'MIW-CE', 'SNIW-CE', 'CGAN-CE', 'Upsampling-CE',
         'IW-MMD', 'MIW-MMD', 'SNIW-MMD', 'Upsampling-MMD'],
        'all', measure=_measure)
except: print('check that df has items')

