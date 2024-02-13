import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from Scripts.setup import macronutrient_dict
import numpy as np
import os
from Scripts.setup import other_nutrients_dict, demographics_dict, macronutrient_dict, essential_vitamins_dict
from tqdm import tqdm

plt.rcParams.update({
    'font.size': 16,
    'lines.linewidth': 4
})

class NhanesDF:

    @staticmethod
    def get_bins_dict():
        d = {
            'income_poverty': [0, 1, 2, 3, 4, 4.99, 5],
            'age_yrs': [0, 2, 13, 20, 30, 40, 50, 60, 70, 79.99, 80]
        }
        return d

    @staticmethod
    def get_spare_to_cat_dict():
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

        d = NhanesDF.get_spare_to_cat_dict()

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
                temp = self.df.loc[self.df[y] < self.df[y].quantile(0.997)]
                sns.boxplot(temp, x=y, y=x, hue=x, ax=ax)
                self.output(save_path)

if __name__ == "__main__":
    d = {}
    d.update(other_nutrients_dict())
    d.update(macronutrient_dict())
    d.update(essential_vitamins_dict())

    p = "/Users/joshfisher/PycharmProjects/MADS_Milestone1/Data/NHANES/nhanes.csv"
    df = pd.read_csv(p)
    hue_vars = list(NhanesDF.get_spare_to_cat_dict().keys())
    hue_vars.remove('married')
    hue_vars.remove('time_in_us')

    df = df.loc[df.year == 2020]
    bins_d = NhanesDF.get_bins_dict()
    temp = 'income_poverty'
    df['poverty_income_binned'] = pd.cut(df[temp], bins_d[temp])
    temp = 'age_yrs'
    df['age_yrs_binned'] = pd.cut(df[temp], bins_d[temp])

    hue_vars.extend(['poverty_income_binned', 'age_yrs_binned'])

    df = NhanesDF.sparse_col_to_str(df, hue_vars)

    save_dir = '/Users/joshfisher/PycharmProjects/MADS_Milestone1/Data/Plots/EDA/Boxplots'
    EDAViz(df).boxplots(hue_vars, list(d.values()), save_dir=save_dir)


