# famous people names
# more than two words

import pandas as pd
import json

# https://pantheon.world/data/datasets
df=pd.read_csv("data/external/person_2020_update.csv", dtype=str)
d = df["name"].tolist()

d = [i.split(",")[0] for i in d]
b = [i.lower() for i in d if len(i.split())>1]


json.dump(
    b,
    open("data/names.json", 'w', encoding='utf-8'),
    indent=4,
    ensure_ascii=False
)
