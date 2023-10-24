import copy
import json
import os
import queue
import threading
import time
from functools import partial

import exp
import numpy as np
import pandas as pd
import ulp
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc


class CustomFloPoAnalyzerKeras:
    def __init__(self, model, file_name_id, get_data_func, string_id):
        self.model = model
        self.file_name_id = file_name_id
        self.ulp_data = []
        self.exp_data = []
        self.string_id = string_id
        self.analysis_data = {}
        ulp_file_path = 'profiling-data/ULP' + '-' + self.string_id + '-' + file_name_id + '.pkl'
        exp_file_path = 'profiling-data/EXP' + '-' + self.string_id + '-' + file_name_id + '.pkl'
        if not os.path.isfile(ulp_file_path) or not os.path.isfile(exp_file_path):
            united_data = get_data_func()
            data_gb = united_data.groupby('layer_name')
            self.splitted_data = [data_gb.get_group(x) for x in data_gb.groups]

    def _analysis(self, data, compute_function, profile_timing, computation_id):
        file_path = 'profiling-data/' + computation_id + '_' + self.string_id + '_' + self.file_name_id + '.pkl'
        if self.file_name_id is not None and os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                data.extend(pickle.load(f))
                return data
        threads = []
        _queue = queue.Queue(len(self.splitted_data))
        # note: it could be further optimize for 2 reasons:
        # 1. not all the layers have the same output dimension
        # 2. number of layers could be less than number of cores
        if profile_timing:
            print(computation_id + ' started')
            ts = time.time()
        for layer_data in self.splitted_data:
            thread = threading.Thread(target=
                                      lambda chunk, name: _queue.put({computation_id: compute_function(chunk),
                                                                      'layer_name': name}),
                                      args=(layer_data['values'].tolist(), layer_data['layer_name'].unique()[0]))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        if profile_timing:
            ts1 = time.time()
            print(computation_id + ' finished. Elapsed time: ' + str(ts1 - ts) + ' s')

        data.extend(list(_queue.queue))
        if self.file_name_id is not None:
            if not os.path.exists('profiling-data'):
                os.makedirs('profiling-data')
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        return data

    def analyze(self, analyze_ulp=True, analyze_exp=True, profile_timing=False):

        ulp_analysis = partial(self._analysis,
                               data=self.ulp_data,
                               compute_function=ulp.compute_nogil,
                               profile_timing=profile_timing,
                               computation_id='ULP')

        exp_analysis = partial(self._analysis,
                               data=self.exp_data,
                               compute_function=exp.compute_nogil,
                               profile_timing=profile_timing,
                               computation_id='EXP')

        if analyze_ulp and analyze_exp:
            return ulp_analysis(), exp_analysis()
        if analyze_ulp and not analyze_exp:
            return ulp_analysis()
        if not analyze_ulp and analyze_exp:
            return exp_analysis()
        else:
            return None

    def mantissa_exponent_analysis(self, min_value_filter_ulp=0.01, min_value_filter_exp=0.01, ulp_percentiles=None):
        if ulp_percentiles is None:
            ulp_percentiles = [35, 40, 50, 60, 70]
        if not self.analysis_data:
            self._generate_df_mantissa_exponent_analysis(min_value_filter_ulp, min_value_filter_exp, ulp_percentiles)
        analysis_data = copy.deepcopy(self.analysis_data)
        for d in analysis_data['layer_data']:
            d.pop('plot_data')
        if not os.path.exists('analysis-report'):
            os.makedirs('analysis-report')
        with open('analysis-report/' + self.string_id + '_' + self.file_name_id + '_PTQ_analysis.json', 'w') as fp:
            json.dump(analysis_data, fp, sort_keys=True, indent=4)
        return analysis_data

    def _generate_df_mantissa_exponent_analysis(self, min_value_filter_ulp, min_value_filter_exp, ulp_percentiles):
        def min_exp_bits_func(min_v, max_v):
            return int(np.ceil(np.log2(max_v - min_v + 1))) if max_v != min_v else 1

        res = \
            {
                'ulp_percentiles': ulp_percentiles,
                'layer_data': []
            }
        for u, e in zip(self.ulp_data, self.exp_data):
            single_res = \
                {
                    'layer_name': '',
                    'plot_data':
                        {
                            'ulps_count': pd.pandas.DataFrame(),
                            'ulps_count_filtered': pd.pandas.DataFrame(),
                            'exps_count': pd.pandas.DataFrame(),
                            'exps_count_filtered': pd.pandas.DataFrame()
                        },
                    'statistical_values':
                        {
                            'min_exp_gt_min': 0,
                            'max_exp_gt_min': 0,
                            'bias_gt_min': 0,
                            'ulp_percentile_values': []
                        },
                    'exact_values':
                        {
                            'min_exp': 0,
                            'max_exp': 0,
                            'exact_bias': 0,
                            'min_exponent_bits': 0,
                            'min_ulp': 0,
                            'min_mantissa_bit': 0
                        }
                }
            df_u = (
                pd.DataFrame(u).
                value_counts(sort=False).
                reset_index().
                rename(columns={2: 'count'}).
                drop(columns=['layer_name'])
            )
            df_u['count_sum_%'] = df_u['count'].cumsum() / len(u['ULP']) * 100
            single_res['plot_data']['ulps_count'] = df_u
            single_res['plot_data']['ulps_count_filtered'] = df_u[df_u['count'] > df_u['count'].max() *
                                                                  min_value_filter_ulp]
            df_e = (
                pd.DataFrame(e).
                value_counts(sort=False).
                reset_index().
                rename(columns={2: 'count'}).
                drop(columns=['layer_name'])
            )
            df_e_filtered = df_e[df_e['count'] > df_e['count'].max() * min_value_filter_exp]
            single_res['plot_data']['exps_count'] = df_e
            single_res['plot_data']['exps_count_filtered'] = df_e_filtered
            single_res['statistical_values']['min_exp_gt_min'] = min(df_e_filtered['EXP'])
            single_res['statistical_values']['max_exp_gt_min'] = max(df_e_filtered['EXP'])
            single_res['statistical_values']['bias_gt_min'] = round(
                (single_res['statistical_values']['max_exp_gt_min'] - single_res['statistical_values'][
                    'min_exp_gt_min'])
                / 2
            )
            single_res['statistical_values']['ulp_percentile_values'].extend(np.percentile(u['ULP'], ulp_percentiles))
            single_res['exact_values']['min_exp'] = min(df_e['EXP'])
            single_res['exact_values']['max_exp'] = max(df_e['EXP'])
            single_res['exact_values']['exact_bias'] = round(
                (single_res['exact_values']['max_exp'] - single_res['exact_values']['min_exp']) / 2
            )
            single_res['exact_values']['min_exponent_bits'] = min_exp_bits_func(single_res['exact_values']['min_exp'],
                                                                                single_res['exact_values']['max_exp'])
            single_res['exact_values']['min_ulp'] = min(df_u['ULP'])
            single_res['exact_values']['min_mantissa_bit'] = int(23 - np.ceil(np.log2(min(df_u['ULP']))))
            single_res['layer_name'] = u['layer_name']
            res['layer_data'].append(single_res)
        self.analysis_data = res
        return res

    def _ulp_plot(self, plot_ulp, plot_cumulative, colors):
        for d in self.analysis_data['layer_data']:
            if plot_ulp:
                fig, ax = plt.subplots(figsize=(16, 9))
                p = sns.barplot(d['plot_data']['ulps_count_filtered'], x='ULP', y='count', ax=ax, lw=0.)
                every_x = (len(ax.get_xticks()) // 50) + 1
                ax.set_xticks(ax.get_xticks()[::every_x])
                p.tick_params(labelrotation=45)
                p.set_title('Distribution of ULP values of ' + self.string_id + ' for layer ' + d['layer_name'])
                p.set_xlabel('ULP [pure number]')
                p.set_ylabel('Number of occurrences [pure number]')
                plt.ylim(
                    round(d['plot_data']['ulps_count_filtered']['count'].min() - d['plot_data']['ulps_count_filtered'][
                        'count'].min() * 0.1),
                    round(d['plot_data']['ulps_count_filtered']['count'].max() + d['plot_data']['ulps_count_filtered'][
                        'count'].max() * 0.1)
                )
                if not os.path.exists(self.model.name + '-plots/'):
                    os.makedirs(self.model.name + '-plots/')
                plt.savefig(self.model.name + '-plots/' + self.string_id + '_' + d['layer_name'] + '_ulp.png', dpi=500)
                plt.close(fig)
            if plot_cumulative:
                fig, ax = plt.subplots(figsize=(16, 9))
                plt.xscale('log', base=2)
                ax.set_xticks([2 ** n for n in np.arange(0, 24)])
                ax.set_yticks([n for n in np.arange(0, 101, 5)])
                p = sns.lineplot(d['plot_data']['ulps_count'], x='ULP', y='count_sum_%', ax=ax)  # , marker='o')
                p.tick_params(labelrotation=45)
                p.set_title('Cumulative Distribution Function of ' + self.string_id + ' for layer ' + d['layer_name'])
                p.set_xlabel('ULP [pure number]')
                p.set_ylabel('ULP Sum Percentage [\%]')

                # Add vertical lines at the specified percentiles
                for percentile, value, color in zip(self.analysis_data['ulp_percentiles'],
                                                    d['statistical_values']['ulp_percentile_values'], colors):
                    # Find the closest data point below the percentile value
                    closest_index = np.argmin(np.abs(d['plot_data']['ulps_count']['ULP'] - value))
                    closest_ulp = d['plot_data']['ulps_count']['ULP'].iloc[closest_index]

                    closest_percentage = d['plot_data']['ulps_count']['count_sum_%'].iloc[closest_index]
                    ax.set_ylim([0, 105])
                    ax.plot([closest_ulp, closest_ulp], [0, closest_percentage], linestyle='--', color=color,
                            label=('{perc}th percentile: $2^{{{val:.2f}}}$'.format(perc=percentile, val=np.log2(value)))
                            )

                # Add a legend to the plot
                ax.legend()
                if not os.path.exists(self.model.name + '-plots/'):
                    os.makedirs(self.model.name + '-plots/')
                plt.savefig(self.model.name + '-plots/' + self.string_id + '_' + d['layer_name'] + '_ulp_cdf.png', dpi=500)
                plt.close(fig)

    def _exp_plot(self):
        for d in self.analysis_data['layer_data']:
            fig, ax = plt.subplots(figsize=(16, 9))
            p = sns.barplot(d['plot_data']['exps_count_filtered'], x='EXP', y='count', ax=ax, lw=0.)
            p.tick_params(labelrotation=45)
            p.set_title('Distribution of Exponent values of ' + self.string_id + ' for layer ' + d['layer_name'])
            p.set_xlabel('Exponent value [pure number]')
            p.set_ylabel('Number of occurrences [pure number]')
            if not os.path.exists(self.model.name + '-plots/'):
                os.makedirs(self.model.name + '-plots/')
            plt.savefig(self.model.name + '-plots/' + self.string_id + '_' + d['layer_name'] + '_exp.png', dpi=500)
            plt.close(fig)

    def make_plots(self, ulp=True, exp=True, cumulative=True, ulp_percentiles=None, colors=None,
                   min_value_filter_ulp=0.01, min_value_filter_exp=0.01):
        if colors is None:
            colors = ['red', 'green', 'orange', 'purple', 'black']
        if ulp_percentiles is None:
            ulp_percentiles = [35, 40, 50, 60, 70]
        if not self.analysis_data:
            self._generate_df_mantissa_exponent_analysis(min_value_filter_ulp, min_value_filter_exp, ulp_percentiles)

        sns.set_style('whitegrid')

        rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)
        if ulp or cumulative:
            self._ulp_plot(ulp, cumulative, colors)
        if exp:
            self._exp_plot()
