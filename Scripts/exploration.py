import os

import pandas as pd
import glob
import re
from tqdm import tqdm
def dir_xpts_to_csv(base_dir, save_p):
    files = os.listdir(base_dir)
    df = pd.DataFrame()

    for file in tqdm(files):
        p = os.path.join(base_dir, file)
        print(p)
        temp = pd.read_sas(p)
        year = int(re.findall(r'-\d+', file)[0].replace('-', ""))
        temp.insert(0, 'year', year)
        df = pd.concat((df, temp))

    df = df.sort_values(['year', 'SEQN'])
    df.to_csv(os.path.join(base_dir, save_p), index=False)

base_dir = '/Data/NHANES/Dietary/IndividualFoods/Day2'
dir_xpts_to_csv(base_dir, 'indiv_food_day2.csv')