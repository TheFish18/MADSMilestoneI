import os

import pandas as pd
import glob
import re
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

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

def essential_vitamins_dict():
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


