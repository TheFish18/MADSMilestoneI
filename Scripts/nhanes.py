import os

import pandas as pd
import glob
import re
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

class NHANESDataFrame:
    """ This class is specific to NHANES dataset and contains cleaning functions, utilities, etc. """
    def __init__(self, path):
        self.path = path

    def clean_demographics_df(self):
        """ returns cleaned, renamed, demographics dataframe """
        demo_d = self.get_demographics_dict()
        df = pd.read_sas(self.path)

        # filter, only include cols in demo_d.keys()
        df = df[list(demo_d.keys())]

        # rename to demo_d.valuese()
        df = df.rename(columns=demo_d)

        # Removing rows where all values are NaN
        temp = list(demo_d.values())
        temp.remove('id')
        return df

    def clean_total_nutrients_df(self):
        """ returns cleaned, renamed, total nutrients dataframe """
        d = {'SEQN': 'id'}
        macro_d = self.get_macronutrient_dict()
        ess_d = self.get_essential_vitamins_dict()
        oth_d = self.get_other_nutrients_dict()

        d.update(macro_d)
        d.update(ess_d)
        d.update(oth_d)

        df = pd.read_sas(self.path)

        # filter
        df = df[list(d.keys())]
        # rename
        df = df.rename(columns=d)
        return df

    @staticmethod
    def read_combine_clean_demographics_total_nutrients(demographics_path, total_nutrients_path):
        """ Reads demographics and total nutrients, clean both, and merge the two dataframe together """
        demo_df = NHANESDataFrame(demographics_path).clean_demographics_df()
        nuts_df = NHANESDataFrame(total_nutrients_path).clean_total_nutrients_df()
        df = demo_df.merge(nuts_df, on='id')
        return df

    @staticmethod
    def get_essential_vitamins_dict():
        """
        https://www.nia.nih.gov/health/vitamins-and-supplements/vitamins-and-minerals-older-adults#:~:text=There%20are%2013%20essential%20vitamins,keep%20the%20body%20working%20properly.
        There are 13 essential vitamins —
        vitamins A, C, D, E, K,
        and the B vitamins (thiamine, riboflavin, niacin, pantothenic acid, biotin, B6, B12, and folate).

        no pantothenic acid in NHANES
        no biotin in NHANES

        :return:
            encoded_var: var
        """
        d = {
            "DR1TVARA": 'vitaminA', "DR1TVC": "vitaminC", "DR1TVD": "vitaminD", "DR1TATOC": 'vitaminE', "DR1TVK": "vitaminK",
            "DR1TVB1": 'thiamine', "DR1TVB2": "riboflavin", "DR1TNIAC": 'niacin', "DR1TVB6": "vitaminB6",
            "DR1TVB12": "vitaminB12", "DR1TFOLA": "folate"
        }
        return d

    @staticmethod
    def get_macronutrient_dict() -> dict:
        """
        Common macronutrients
        :return:
            encoded_var: var
        """
        d = {"DR1TKCAL": "kcals", "DR1TPROT": "protein", "DR1TCARB": "carbs", "DR1TSUGR": "sugar",
            "DR1TTFAT": "total_fat", "DR1TSFAT": "sat_fat", "DR1TMFAT": "mono_fat", "DR1TPFAT": "poly_fat",
            "DR1TCHOL": "cholesterol"}
        return d

    @staticmethod
    def get_other_nutrients_dict() -> dict:
        d = {"DR1TCALC": "calcium", "DR1TPHOS": "phosphorus", "DR1TMAGN": "magnesium",
             "DR1TIRON": "iron", "DR1TZINC": "zinc", "DR1TCOPP": "copper", "DR1TSODI": "sodium",
             "DR1TPOTA": "potassium", "DR1TSELE": "selenium", "DR1TCAFF": "caffeine",
             "DR1TTHEO": "theobromine", "DR1TALCO": "alcohol"}
        return d

    @staticmethod
    def get_demographics_dict() -> dict:
        with open('../Data/Final/demographics.json') as f:
            d = json.load(f)
        return d

    @staticmethod
    def get_nhanes_to_fda_rec_dict() -> dict:

        """
        d was gotten using:
            cols = df_nhanes.columns.tolist()
            recs = df_fda.Nutrient.unique().tolist()
            recs_ = [x.lower().replace(" ", "_") for x in recs]
            d = {c: recs[recs_.index(c)] for c in cols if c in recs_}

        add_vitamins was gotten using:
            cols = df_nhanes.columns.tolist()
            recs = df_fda.Nutrient.unique().tolist()
            recs = [x for x in recs if " " in x]
            recs_ = [x.split(" ") for x in recs if " " in x]
            recs_ = [x[0].lower() + x[1].upper() for x in recs_]
            d = {c: recs[recs_.index(c)] for c in cols if c in recs_}
        """
        d = {'protein': 'Protein', 'cholesterol': 'Cholesterol', 'riboflavin': 'Riboflavin', 'niacin': 'Niacin',
             'calcium': 'Calcium', 'phosphorus': 'Phosphorus', 'magnesium': 'Magnesium', 'iron': 'Iron',
             'zinc': 'Zinc', 'copper': 'Copper', 'sodium': 'Sodium', 'potassium': 'Potassium', 'selenium': 'Selenium'}

        add_vitamins = {
            'vitaminE': 'Vitamin E', 'vitaminA': 'Vitamin A', 'vitaminB6': 'Vitamin B6',
            'vitaminB12': 'Vitamin B12', 'vitaminC': 'Vitamin C', 'vitaminD': 'Vitamin D', 'vitaminK': 'Vitamin K'}

        d.update(add_vitamins)
        return d

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

        d = NHANESDataFrame.get_sparse_to_cat_dict()

        if type(cols) == str:
            cols = [cols]
        if type(cols) == list:
            mapped_keys = set(d.keys())
            if len(set(cols) - mapped_keys) > 0:
                print(f"Dont have mappings for: {set(cols) - mapped_keys}")
            d = {k: v for k, v in d.items() if k in cols}

        new_df = df.replace(d)
        return new_df

    @staticmethod
    def get_demographics_df():
        """ Convenient function to get cleaned demographics dataframes"""
        df = pd.read_csv("../Data/Final/demographics_clean.csv")
        return df

    @staticmethod
    def get_total_nutrients_df():
        """ Convenient function to get cleaned total nutrients dataframes"""
        df = pd.read_csv("../Data/Final/total_nutrients_clean.csv")
        return df

    @staticmethod
    def get_combined_df():
        """ Convenient function to get cleaned and combined demographics and total nutrient file """
        df = pd.read_csv('../Data/Final/nhanes.csv')
        return df.loc[df.year == 2020]


def dir_xpts_to_csv(base_dir, save_p):
    """
    Takes a folder of XPTs and converts to a single csv
    :param base_dir: Folder path
    :param save_p: Where to save output csv
    :return: None
    """
    files = os.listdir(base_dir)
    df = pd.DataFrame()

    for file in tqdm(files):
        p = os.path.join(base_dir, file)
        temp = pd.read_sas(p)
        year = int(re.findall(r'-\d+', file)[0].replace('-', ""))
        print(year)
        temp.insert(0, 'year', year)
        df = pd.concat((df, temp))

    df = df.sort_values(['year', 'SEQN'])
    df.to_csv(os.path.join(base_dir, save_p), index=False)

def read_filter_rename(df_p, data_dict=None, data_dict_p=None) -> pd.DataFrame:
    """
    Read in a df, filter columns, and rename. Provide either data_dict or data_dict_p
    :param df_p: path to dataframe
    :param data_dict: {original col name: new col name}
    :param data_dict_p: path to json of above
    :return: pd.DataFrame
    """
    if data_dict is None:
        assert data_dict_p is not None, "Must either provide data_dict or data_dict_p"
        with open(data_dict_p) as f:
            data_dict = json.load(f)

    df = pd.read_csv(df_p)
    df = df[list(data_dict.keys())].rename(columns=data_dict)
    return df

def get_individual_foods_col_names(url=None):
    if url is None:
        url = "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_DR1IFF.htm#Codebook"
    cols = pd.read_html(url)[-2]
    cols = cols[['Day1 Name', "Variable Label"]].set_index('Day1 Name')
    cols = cols.to_json()
    cols = json.loads(cols)
    return cols["Variable Label"]

def get_total_nutrients_col_names(url=None):
    if url is None:
        url = "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/P_DR1TOT.htm"
    cols = pd.read_html(url)[-1]
    cols = cols[['Day1 Name', "Variable Label"]].set_index('Day1 Name')
    cols = cols.to_json()
    cols = json.loads(cols)
    return cols["Variable Label"]

def get_common_deficiencies():
    """
    srcs:
    https://www.health.harvard.edu/nutrition/the-truth-about-nutrient-deficiencies
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9710417/
    :return:
    """
    l = ["Vitamin D", "Iron", "Vitamin B12", "Calcium", ]
    raise NotImplementedError

def essential_vitamins_dict() -> dict:
    """
    https://www.nia.nih.gov/health/vitamins-and-supplements/vitamins-and-minerals-older-adults#:~:text=There%20are%2013%20essential%20vitamins,keep%20the%20body%20working%20properly.
    There are 13 essential vitamins —
    vitamins A, C, D, E, K,
    and the B vitamins (thiamine, riboflavin, niacin, pantothenic acid, biotin, B6, B12, and folate).

    no pantothenic acid
    no biotin

    :return: dict of col names
    """
    d = {
        "DR1TVARA": 'vitaminA', "DR1TVC": "vitaminC", "DR1TVD": "vitaminD", "DR1TATOC": 'vitaminE', "DR1TVK": "vitaminK",
        "DR1TVB1": 'thiamine', "DR1TVB2": "riboflavin", "DR1TNIAC": 'niacin', "DR1TVB6": "vitaminB6",
        "DR1TVB12": "vitaminB12", "DR1TFOLA": "folate"
    }
    return d

def macronutrient_dict():
    d = {"DR1TKCAL": "kcals", "DR1TPROT": "protein", "DR1TCARB": "carbs", "DR1TSUGR": "sugar",
        "DR1TTFAT": "total_fat", "DR1TSFAT": "sat_fat", "DR1TMFAT": "mono_fat", "DR1TPFAT": "poly_fat",
        "DR1TCHOL": "cholesterol"}
    return d

def other_nutrients_dict():
    d = {"DR1TCALC": "calcium", "DR1TPHOS": "phosphorus", "DR1TMAGN": "magnesium",
         "DR1TIRON": "iron", "DR1TZINC": "zinc", "DR1TCOPP": "copper", "DR1TSODI": "sodium",
         "DR1TPOTA": "potassium", "DR1TSELE": "selenium", "DR1TCAFF": "caffeine",
         "DR1TTHEO": "theobromine", "DR1TALCO": "alcohol"}
    return d

def demographics_dict():
    with open('/Users/joshfisher/PycharmProjects/MADS_Milestone1/Data/NHANES/Demographic/data_dict.json') as f:
        d = json.load(f)

    return d

def clean_demographics(p):
    """
    ppl born in US: time in us == -1
    :param p:
    :return:
    """
    demo_d = demographics_dict()
    df = pd.read_csv(p, usecols=list(demo_d.keys())).rename(columns=demo_d)

    temp = list(demo_d.values())
    temp.remove("id")
    temp.remove("year")

    df = df.dropna(subset=temp, how='all')

    return df

def clean_total_nutrients(p):
    d = {'SEQN': 'id', 'year': 'year'}
    macro_d = macronutrient_dict()
    ess_d = essential_vitamins_dict()
    oth_d = other_nutrients_dict()

    d.update(macro_d)
    d.update(ess_d)
    d.update(oth_d)

    df = pd.read_csv(p, usecols=list(d.keys())).rename(columns=d)

    temp = list(d.values())
    temp.remove("id")
    temp.remove("year")

    df = df.dropna(subset=temp, how='all')
    return df

def combine_clean_demo_totnut(demo_p, tot_nut_p):
    demo_df = pd.read_csv(demo_p)
    nut_df = pd.read_csv(tot_nut_p)

    demo_df = demo_df.drop(columns=['year'])
    df = demo_df.merge(nut_df, on='id')
    return df

df = pd.read_csv("/Users/joshfisher/PycharmProjects/MADS_Milestone1/Data/NHANES/nhanes.csv")


