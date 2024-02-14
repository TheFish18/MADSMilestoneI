import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Scripts.setup import macronutrient_dict
import numpy as np
import os
from Scripts.setup import other_nutrients_dict, demographics_dict, macronutrient_dict, essential_vitamins_dict
from tqdm import tqdm
from scipy.stats import zscore
import sys


# plt.rcParams.update({
#     'font.size': 16,
#     'lines.linewidth': 4
# })


def ridgeplot(df, standardize=True, units="", xlim=None, sorder=None):
    """
    creates a ridgeplot from the given dataframe

    parmas:
        df (pandas.DataFrame): a pandas dataframe with at least two numeric features
        standardize (bool): performs z-score normalization
        units (string): optionally specify units
        xlim (tuple): optionally specify limits of x axis

    returns:
        None and outputs a ridgeplot

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
    # TODO: dont hardcode number for palette

    g = sns.FacetGrid(df, row='variable', hue='variable', palette=palette, row_order=sorder, hue_order=sorder,
                      aspect=20, height=1, sharex=True, xlim=xlim)

    g.map_dataframe(sns.kdeplot, x='value', fill=True, alpha=1)
    g.map(label, 'variable')
    g.map_dataframe(sns.kdeplot, x='value', color='black', alpha=1)
    g.fig.subplots_adjust(hspace=-0.6)
    g.set_titles("")
    g.set_axis_labels(y_var="", x_var=x_var)
    g.set(yticks=[])
    g.despine(left=True)
    plt.show()


class NhanesDF:

    @staticmethod
    def get_bins_dict():
        d = {
            'income_poverty': [0, 1, 2, 3, 4, 4.99, 5],
            'age_yrs': [0, 2, 13, 20, 30, 40, 50, 60, 70, 79.99, 80]
        }
        return d

    @staticmethod
    def get_sparse_to_cat_dict():
        d = {
            'gender': {1: "Male", 2: "Female"},
            'race': {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
                     4: "Non-Hispanic Black", 5: "Other"},
            'education': {1: "<9th Grade", 2: "9-11th Grade", 3: "High School", 4: "Some College",
                          5: ">College", 7: "Refused", 9: "Don\'t Know"},
            'pregnant': {1: "Pregnant", 2: "Not Pregnant", 3: "Uncertain"},
            'country_born': {1: "USA", 2: "Other", 77: "Refused", 99: "Don\'t Know"},
            'married': {1: "Married", 2: "Separated", 3: "Never Married", 77: "Refused", 99: "Don\'t Know"},
            'time_in_us': {1: "<5 Yrs", 2: "5-15 Yrs", 3: "15-30 Yrs", 4: ">30 Yrs", 77: "Refused", 99: "Don\'t Know"}
        }
        return d

    @staticmethod
    def sparse_col_to_str(df, cols=None):
        """
        Converts a sparse encoded col(s) to str representation. Uses rename_dict for mapping
        :param cols:
            - If None: uses all keys in rename_dict()
            - If list of col names: uses those column names
            - If string: use single col name
        :return:
            NhanesDF with new df vals
        """
        assert cols is None or type(cols) == list or type(cols) == str, "cols must either be None, list, or str"

        d = NhanesDF.get_sparse_to_cat_dict()

        if type(cols) == str:
            cols = [cols]
        if type(cols) == list:
            mapped_keys = set(d.keys())
            if len(set(cols) - mapped_keys) > 0:
                print(f"Dont have mappings for: {set(cols) - mapped_keys}")
            d = {k: v for k, v in d.items() if k in cols}

        new_df = df.replace(d)
        return new_df

class EDAViz:
    def __init__(self, nhanes_df):
        self.df = nhanes_df

    def output(self, save_p=None):
        if save_p is None:
            plt.show()
        else:
            plt.savefig(save_p)
        plt.close()

    def plot_line_year_ys_groupbys(self, y_vars, groupby_vars, x='year', save_dir=None, func='mean'):
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
            temp = self.df.groupby([x, groupby_var]).agg(func).reset_index()
            temp = NhanesDF.sparse_col_to_str(temp, groupby_var)
            for y_var in y_vars:
                f, ax = plt.subplots(figsize=(15, 15))
                sns.lineplot(temp, x=x, y=y_var, hue=groupby_var, ax=ax)
                plt.tight_layout()

                if save_dir is not None:
                    save_dir = os.path.join(save_dir, y_var, f"{x}_{y_var}_{groupby_var}.png")

                self.output(save_dir)

    def boxplots(self, xs, ys, save_dir=None):
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
                self.output(save_path)

    @staticmethod
    def ridgeplot(vars=None):
        if vars is None:
            vars = list(other_nutrients_dict().values())
            vars.remove('alcohol')
            vars.remove('copper')
            vars.remove('phosphorus')
            vars.remove('potassium')
            vars.remove('caffeine')
            vars.remove('theobromine')
        vars = ['caffeine', 'vitaminD', 'vitaminC', *vars]
        print(vars)
        p = "/Users/joshfisher/PycharmProjects/MADS_Milestone1/Data/NHANES/nhanes.csv"
        df = pd.read_csv(p)

        df = df.loc[df.year == 2020]
        df = df[vars]

        ridgeplot(df, xlim=(-2, 2), sorder=vars)

    @ staticmethod
    def iron_sodium_joint_grid():
        p = "/Users/joshfisher/PycharmProjects/MADS_Milestone1/Data/NHANES/nhanes.csv"
        df = pd.read_csv(p)
        hue_vars = list(NhanesDF.get_sparse_to_cat_dict().keys())
        hue_vars.remove('married')
        hue_vars.remove('time_in_us')

        df = df.loc[df.year == 2020]

        g = sns.JointGrid(data=df, x="iron", y='sodium', space=.05, xlim=(0, 50), ylim=(0, 10000))
        g.plot_joint(
            sns.kdeplot, fill=True, cmap='rocket', clip=((0, 50), (0, 10000)), thresh=0, levels=100)
        g.plot_marginals(sns.histplot, color='#03051A', alpha=1, bins=100)
        plt.show()

    @staticmethod
    def z_test():
        from scipy.stats import ttest_ind
        """
        Two sided T test:
        H0: Two samples have identical means
        Ha: Means differ 
        """
        p = "/Users/joshfisher/PycharmProjects/MADS_Milestone1/Data/NHANES/nhanes.csv"
        df = pd.read_csv(p)

        male_df = df.loc[df.gender == 1]
        female_df = df.loc[df.gender == 2]

        male_cal_standard = male_df.calcium / male_df.kcals
        female_cal_standard = female_df.calcium / female_df.kcals

        res = ttest_ind(male_cal_standard, female_cal_standard)
        print(res)

    @staticmethod
    def violin_plot_calcium():
        p = "/Users/joshfisher/PycharmProjects/MADS_Milestone1/Data/NHANES/nhanes.csv"
        df = pd.read_csv(p)
        hue_vars = list(NhanesDF.get_sparse_to_cat_dict().keys())
        hue_vars.remove('married')
        hue_vars.remove('time_in_us')

        df = NhanesDF.sparse_col_to_str(df)
        df = df.loc[df.year == 2020]
        # df = df.loc[df.age_yrs > 50]
        df['age_yrs_binned'] = pd.cut(df['age_yrs'], [0, 20, 50, 79.9, 150], labels=["0-20", "20-50", "50-80", "80+"])

        f, ax = plt.subplots(figsize=(12, 12))
        palette = sns.color_palette('flare', 5)
        palette = palette[0], palette[4]
        # sns.boxplot(data=df, x='calcium', y='age_yrs_binned', hue='gender', ax=ax)
        # sns.stripplot(data=df, x='calcium', y='age_yrs_binned', hue='gender', ax=ax, dodge=True, jitter=0.25)
        sns.violinplot(data=df, x='age_yrs_binned', y='calcium', hue='gender',
                       split=True, gap=0.1, inner='quart', palette=palette, ax=ax)
        plt.ylim((-100, 3500))
        plt.show()

if __name__ == "__main__":

    EDAViz.violin_plot_calcium()
    sys.exit()
    d = {}
    d.update(other_nutrients_dict())
    d.update(macronutrient_dict())
    d.update(essential_vitamins_dict())

    f, ax = plt.subplots(figsize=(12, 12))

    bins_d = NhanesDF.get_bins_dict()

    p = "/Users/joshfisher/PycharmProjects/MADS_Milestone1/Data/NHANES/nhanes.csv"
    df = pd.read_csv(p)
    hue_vars = list(NhanesDF.get_sparse_to_cat_dict().keys())
    hue_vars.remove('married')
    hue_vars.remove('time_in_us')

    df = NhanesDF.sparse_col_to_str(df)
    df = df.loc[df.year == 2020]
    # df = df.loc[df.age_yrs > 50]
    df['age_yrs_binned'] = pd.cut(df['age_yrs'], [0, 20, 50, 79.9, 150], labels=["0-20", "20-50", "50-80", "80+"])

    # sns.boxplot(data=df, x='calcium', y='age_yrs_binned', hue='gender', ax=ax)
    # sns.stripplot(data=df, x='calcium', y='age_yrs_binned', hue='gender', ax=ax, dodge=True, jitter=0.25)
    sns.violinplot(data=df, x='age_yrs_binned', y='calcium', hue='gender', split=True, gap=0.1, inner='quart', ax=ax)
    plt.ylim((-100, 3500))
    plt.show()
    #
    # sns.kdeplot(data=df, x='calcium', hue='gender')
    # plt.show()

    # print(essential_vitamins_dict())
    # df = df[list(essential_vitamins_dict().values())]
    # print(len(df.columns))
    # ridgeplot(df)
    # temp = 'income_poverty'
    # df['poverty_income_binned'] = pd.cut(df[temp], bins_d[temp])
    # temp = 'age_yrs'
    #
    # hue_vars.extend(['poverty_income_binned', 'age_yrs_binned'])
    #
    # keep_cols = [*hue_vars, *list(essential_vitamins_dict().values())]
    # df = NhanesDF.sparse_col_to_str(df, hue_vars)[keep_cols]
    #
    # vars_ = []
    # for vit in list(essential_vitamins_dict().values()):
    #     v = f'{vit}_deficient'
    #     vars_.append(v)
    #     df[v] = df[vit] < df[vit].quantile(.10)
    #
    # df['total_deficiencies'] = df[vars_].sum(axis=1)
    # df = df[[*hue_vars, 'total_deficiencies']]
    #
    # print(df.head().to_string())
    #
    # # print(df.groupby(['race']).value_counts(['total_deficiencies'], True).to_string())
    # save_dir = '/Users/joshfisher/PycharmProjects/MADS_Milestone1/Data/Plots/EDA/Boxplots/Deficiences'
    # EDAViz(df).boxplots(hue_vars, ['total_deficiencies'], save_dir=save_dir)


