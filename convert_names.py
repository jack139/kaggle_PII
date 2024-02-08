import pandas as pd
import json

df=pd.read_csv("data/external/pantheon.tsv", sep='\t')
d = df["name"].tolist()
b = [i for i in d if len(i.split())>1]
b += ['Jeff Bezos']

json.dump(
    b,
    open("data/names.json", 'w', encoding='utf-8'),
    indent=4,
    ensure_ascii=False
)
