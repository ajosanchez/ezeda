import math
import logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Plotter():
    def __init__(self, df):
        self.df = df
        self.numeric = [c for c in df if pd.api.types.is_numeric_dtype(df[c].dtype)]
        self.categorical = [c for c in df if isinstance(df[c].dtype, pd.CategoricalDtype)]
        logging.info("%d numeric and %d categorical column(s) found", len(self.numeric), len(self.categorical))

    def set_cols(self, arg):
        if arg is None:
            return self.numeric + self.categorical

        if isinstance(arg, str) or isinstance(arg, int):
            cols = [arg]
        else:
            cols = arg

        return [self.df.columns[i] if isinstance(i, int) else i for i in cols]

    def plot(self, cols=None, target=None, figsize=None):
        cols = self.set_cols(cols)
        numeric_plots = list(set(cols) & set(self.numeric))
        categorical_plots = list(set(cols) & set(self.categorical))
        # number of axs may be > number of plots because new plot types should start on a new row of the grid
        num_numeric_axs, num_categorical_axs = self.get_num_axs(len(numeric_plots), len(categorical_plots))
        # the dims will depend on number of plots and what plots they are
        if len(cols) == 1:
            figsize = figsize or self.get_figsize(num_numeric_axs, num_categorical_axs)
        else:
            figsize = self.get_figsize(num_numeric_axs, num_categorical_axs)
        total_axs = num_numeric_axs + num_categorical_axs
        subplots = numeric_plots + ([None] * (num_numeric_axs-len(numeric_plots))) + categorical_plots + ([None]*(num_categorical_axs-len(categorical_plots)))
        num_plot_rows = math.ceil(total_axs / 3)
        num_plot_cols = min(total_axs, 3)
        height_ratios = ([1]*math.ceil(num_numeric_axs/3)) + ([2]*math.ceil(num_categorical_axs/3))

        fig, axs = plt.subplots(
            num_plot_rows, 
            num_plot_cols, 
            figsize=figsize,
            gridspec_kw={'height_ratios': height_ratios}
        )
        
        if isinstance(axs, plt.Subplot):
            axs = np.array(axs)
        for i, ax in enumerate(axs.flatten()):
            col = subplots.pop(0)
            if col is None:
                continue
            else:
                title = f"{col} ({list(self.df.columns).index(col)})"
                if i < num_numeric_axs:
                    data = self.wrangle_numeric_data(col, target)
                    self.make_hist(data, target, ax=ax, title=title)
                else:
                    data = self.wrangle_categorical_data(col, target)
                    self.make_barh(data, ax=ax, title=title)
        plt.tight_layout()

    def make_hist(self, data, target, **kwargs):
        bins = 10
        ax = data.plot(kind='hist', log=True, alpha=0.8, **kwargs)
        if target is not None:
            patches = ax.patches[:bins]
            edges = [p.get_x() for p in patches] + [patches[-1].get_x() + patches[-1].get_width()]
            bins_0 = [p for p in ax.patches[:bins]]
            bins_1 = [p for p in ax.patches[bins:]]
            heights = np.array([b.get_height() for b in bins_0]) + np.array([b.get_height() for b in bins_1])
            base_ratio = self.df[target].value_counts(normalize=True).loc[1]
            for bin_start, bin_end, bin_height in zip(edges[:-1], edges[1:], heights):
                center = (bin_start + bin_end) / 2
                threshold = bin_height * base_ratio
                ax.plot(center, threshold, marker='x', markersize=5, color='black')
            
    def make_barh(self, data, **kwargs):
        if isinstance(data, pd.Series):
            data.plot(kind='barh', stacked=True, alpha=0.8, **kwargs)
        else:
            ratios = data[1].div(data.sum(1)).round(3)
            ax = data.plot(kind='barh', stacked=True, alpha=0.8, **kwargs)
            ax.bar_label(ax.containers[1], ratios, label_type='center')

    def wrangle_numeric_data(self, col, target):
        if target is None:
            return self.df[col]
        else:
            return self.df.pivot(columns=target, values=col)

    def wrangle_categorical_data(self, col, target):
        if target is None:
            return self.df[col].value_counts()
        else:
            return self.df.pivot(columns=target, values=col).apply(lambda x: x.value_counts())

    def get_num_axs(self, num_numeric_plots, num_categorical_plots):
        if num_categorical_plots % 3 == 0:
            num_categorical_axs = num_categorical_plots
        else:
            if num_numeric_plots == 0 and num_categorical_plots < 3:
                num_categorical_axs = num_categorical_plots
            else:
                num_categorical_axs = num_categorical_plots + (3 - (num_categorical_plots % 3))

        if num_numeric_plots % 3 == 0:
            num_numeric_axs = num_numeric_plots
        else:
            if num_categorical_plots == 0 and num_numeric_plots < 3:
                num_numeric_axs = num_numeric_plots
            else:
                num_numeric_axs = num_numeric_plots + (3 - (num_numeric_plots % 3))

        return (num_numeric_axs, num_categorical_axs)
        
    def get_figsize(self, num_numeric_axs, num_categorical_axs):
        total_axs = num_categorical_axs + num_numeric_axs
        categorical_height = 6 * math.ceil(num_categorical_axs / 3)
        numeric_height     = 3 * math.ceil(num_numeric_axs     / 3)
        fig_height = categorical_height + numeric_height
        fig_width = min(5 * (total_axs), 15)
        return (fig_width, fig_height)
        
