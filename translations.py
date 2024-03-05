from src.utils.translator import translate_texts
import pandas as pd
from tqdm import tqdm

lang_dict = {"amh" : "am", 
                          "afr": "af", 
                          "arb": "ar", 
                          "ary": "ar", 
                          "arq": "ar",
                          "esp": "es", 
                          "eng": "en", 
                          "hau": "ha", 
                          "hin": "hi", 
                          "ind" : "id",
                          "kin": "rw", 
                          "pan": "pa", 
                          "tel": "te",
                          "mar": "mr",
                          "de": "de"}

language = "arq"
input_csv = "res\\data\\a\\arq\\arq_train.csv"
output_path = "res\\data\\a\\arq\\arq_train_translations.csv"
batch_size = 64

raw_df = pd.read_csv(input_csv, encoding="utf-8")
xs1, xs2 = map(list, zip(*[tuple(row['Text'].split('\n')) for idx, row in raw_df.iterrows()]))
s1s, s2s = [], []
for i in (pbar := tqdm(range(len(xs1) // batch_size + 1))):
    pbar.set_description("Translation")
    batch_s1 = xs1[i * batch_size: (i + 1) * batch_size]
    batch_s2 = xs2[i * batch_size: (i + 1) * batch_size]
    batch_s1 = translate_texts(batch_s1, lang_dict[language], lang_dict["eng"])
    batch_s2 = translate_texts(batch_s2, lang_dict[language], lang_dict["eng"])
    s1s.extend(batch_s1)
    s2s.extend(batch_s2)


data = []
for i in range(len(xs1)):
    s1 = xs1[i]
    s2 = xs2[i]
    t1 = s1s[i]
    t2 = s2s[i]
    instance = {"Text1": s1, "Text1 Translation": t1, "Text2": s2, "Text2 Translation": t2}
    data.append(instance)

df = pd.DataFrame(data)
df["PairID"] = raw_df["PairID"]
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
df.to_csv(output_path, encoding="utf-8", index=None)