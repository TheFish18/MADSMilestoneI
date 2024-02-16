import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Scripts.nhanes import macronutrient_dict
import numpy as np
import os
from Scripts.nhanes import other_nutrients_dict, demographics_dict, macronutrient_dict, essential_vitamins_dict
from tqdm import tqdm
from scipy.stats import zscore
import sys


# plt.rcParams.update({
#     'font.size': 16,
#     'lines.linewidth': 4
# })

class Visualizations:
    """ More advanced reusable graphics """
    @staticmethod
    def output(save_p: str = None):
        """ saves fig or shows fig if save_p is None, clears fig afterwards """
        if save_p is None:
            plt.show()
        else:
            plt.savefig(save_p)

        plt.close()

    @staticmethod
    def ridgeplot(df, standardize=True, units="", xlim=None, sorder=None, save_p=None):
        """
        creates a ridgeplot from the given dataframe

        :param df: (pd.DataFrame) with at least two numeric features
        :param standardize: (bool) True: peforms z-score normalization
        :param units: (string) optionally specify units
        :param xlim: (string) optionally specify x limit
        :param sorder: (string) optionally specify output order
        :param save_p: (string) optionally specify whether/where to save
        :return: None
        """
        def process_data(df, standardize):
            if standardize:
                return zscore(df.select_dtypes(include=np.number)).melt()
            else:
                return df.select_dtypes(include=np.number).melt()

        def get_sorder(df):  # rough sort
            temp = zscore(df.select_dtypes(include=np.number)).melt()
            sorder = temp.loc[(temp.value < 0.5) & (temp.value > -0.5)]['variable'].value_counts().index
            return sorder

        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .2, label, color='black', fontsize=13,
                    ha="left", va="center", transform=ax.transAxes)

        sorder = get_sorder(df) if sorder is None else sorder
        num_of_cols = len(df.columns)
        df = process_data(df, standardize)
        x_var = "z-score" if standardize else units
        xlim = (-5, 5) if xlim is None else xlim

        sns.set_theme(style='white', rc={'axes.facecolor': (0, 0, 0, 0), 'axes.linewidth': 2})
        palette = sns.color_palette('flare', num_of_cols)[::-1]

        plt.figure(figsize=(12, 12))
        g = sns.FacetGrid(df, row='variable', hue='variable', palette=palette, row_order=sorder, hue_order=sorder,
                          aspect=12, height=1, sharex=True, xlim=xlim)

        g.map_dataframe(sns.kdeplot, x='value', fill=True, alpha=1)
        g.map(label, 'variable')
        g.map_dataframe(sns.kdeplot, x='value', color='black', alpha=1)
        g.fig.subplots_adjust(hspace=-0.6)
        g.set_titles("")
        g.set_axis_labels(y_var="", x_var=x_var)
        g.set(yticks=[])
        g.despine(left=True)

        Visualizations.output(save_p)

    @staticmethod
    def joint_grid_2dkde_marg_hist(df, x, y, xlim=None, ylim=None, save_p=None):
        """
        Plots a joint grid with 2d kde in the interior and histograms on the margins
        :param df: pd.DataFrame: that contains x and y as cols
        :param x: str: name of x column
        :param y: str: name of y column
        :param xlim: tuple: (x_min, x_max)
        :param ylim: tuple: (y_min, y_max)
        :param save_p: path or None:
            if path: saves to path
            if None: shows fig
        :return:
        """
        g = sns.JointGrid(data=df, x=x, y=y, space=.05, xlim=xlim, ylim=ylim)
        g.plot_joint(
            sns.kdeplot, fill=True, cmap='rocket', clip=(xlim, ylim), thresh=0, levels=100)
        g.plot_marginals(sns.histplot, color='#03051A', alpha=1, bins=100)

        Visualizations.output(save_p)

    @staticmethod
    def violin_plot_calcium(df, x, y, hue, ylim=None, save_p=None):
        """
        Creates a split violin plot
        :param df: pd.DataFrame
        :param x: x column, must be in df
        :param y: y column, must be in df
        :param hue: hue categorical variable, must be in df
        :param ylim: (y_min, y_max)
        :param save_p: path or None:
            if path: saves to path
            if None: shows fig
        :return:
        """
        f, ax = plt.subplots(figsize=(12, 12))
        palette = sns.color_palette('flare', 5)
        palette = palette[0], palette[4]
        sns.violinplot(data=df, x=x, y=y, hue=hue,
                       split=True, gap=0.1, inner='quart', palette=palette, ax=ax)
        plt.ylim(ylim)
        Visualizations.output(save_p)

class AutoEDAVisualizatons:

    @ staticmethod
    def boxplots(df, xs, ys, save_dir=None):
        save_path = None
        for x in tqdm(xs):
            if save_dir is not None:
                os.makedirs(os.path.join(save_dir, x), exist_ok=True)
            for y in ys:
                f, ax = plt.subplots(figsize=(12, 12))
                if save_dir is not None:
                    save_path = os.path.join(save_dir, x, f'{x}_{y}.png')
                # temp = self.df.loc[self.df[y] < self.df[y].quantile(0.997)]
                sns.boxplot(df, x=y, y=x, hue=x, ax=ax)
                Visualizations.output(save_path)

    @staticmethod
    def plot_line_year_ys_groupbys(df, y_vars, groupby_vars, x='year', save_dir=None, func='mean'):
        """
        Plots a series of year, y, groupby plots
        :param y_vars: list of variable names to plot as y value
        :param groupby_vars:
        :param x:
        :param save_dir:
        :param func:
        :return:
        """
        for groupby_var in groupby_vars:
            temp = df.groupby([x, groupby_var]).agg(func).reset_index()
            for y_var in y_vars:
                f, ax = plt.subplots(figsize=(15, 15))
                sns.lineplot(temp, x=x, y=y_var, hue=groupby_var, ax=ax)
                plt.tight_layout()

                if save_dir is not None:
                    save_dir = os.path.join(save_dir, y_var, f"{x}_{y_var}_{groupby_var}.png")

                Visualizations.output(save_dir)

if __name__ == "__main__":
    from nhanes import NHANESDataFrame

    def plot_iron_sodium_joint_grid():
        """ plots a iron sodium joint grid. Full code to plot 2d kde joint gride can be found at visualizations.Visualizations.joint_grid_2dkde_marg_hist """
        df = NHANESDataFrame.get_combined_df()
        x = 'Iron (mg)'
        y = 'Sodium (mg)'

        d = {'iron': x, "sodium": y}
        df = df.rename(columns=d)

        xlim = (0, 50)
        ylim = (0, 10000)
        Visualizations().joint_grid_2dkde_marg_hist(df, x, y, xlim, ylim, "/Users/joshfisher/PycharmProjects/MADS_Milestone1/Data/Final/Plots/Writeup/jointgrid.png")


    plot_iron_sodium_joint_grid()
