import requests
from src.smatch.smatch import get_amr_match, compute_f
from src.utils.translator import translate_texts
from tqdm import tqdm
import pandas as pd
import os
class AMR(object):
    """
    adapted from https://github.com/semantic-textual-relatedness/Semantic_Relatedness_SemEval2024/blob/main/STR_Baseline.ipynb
    """
    def __init__(self, config, src_lang, tgt_lang, batch_size=32):
        super(AMR, self).__init__()
        self.source_language = config.tgt_lan
        self.target_langauge = config.translate_lang
        self.batch_size = batch_size
        self.config = config
        self.lang_dict = {"amh" : "am", 
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

    def smatch_score(self, s1, s2):
        amr1 = self.__get_amr(s1)
        amr2 = self.__get_amr(s2)
        scores = get_amr_match(amr1, amr2)
        p, r, f = compute_f(scores[0], scores[1], scores[2])
        return f, amr1, amr2

    def predict(self, s1s, s2s):
        if self.source_language != "eng":
            scores = []
            data = []
            s1s_temp = s1s
            s2s_temp = s2s
            s1s, s2s = [], []
            for i in (pbar := tqdm(range(len(s1s_temp) // self.batch_size + 1))):
                pbar.set_description("Translation")
                batch_s1 = s1s_temp[i * self.batch_size: (i + 1) * self.batch_size]
                batch_s2 = s2s_temp[i * self.batch_size: (i + 1) * self.batch_size]
                batch_s1 = translate_texts(batch_s1, self.lang_dict[self.source_language], self.lang_dict[self.target_langauge])
                batch_s2 = translate_texts(batch_s2, self.lang_dict[self.source_language], self.lang_dict[self.target_langauge])
                s1s.extend(batch_s1)
                s2s.extend(batch_s2)
        
        for i, pair in enumerate(tqdm(zip(s1s, s2s))):
            s1, s2 = pair
            f, amr1, amr2 = self.smatch_score(s1, s2)
            if self.source_language != "eng":
                s1_orig, s2_orig = s1s_temp[i], s2s_temp[i]
                data.append({"s1": s1_orig, "s1_translation": s1, "amr1": amr1, "s2": s2_orig, "s2_translation": s2, "amr2": amr2, "score": f})
            else:
                data.append({"s1": s1, "amr1": amr1, "s2": s2, "amr2": amr2, "score": f})
            scores.append(f)
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.config.RESULTS_PATH, "metadata.csv"), index=None)
        return scores
    
    def __get_amr(self, s):
        params= {"sentence": s}
        r = requests.get("https://nlp.uniroma1.it/spring/api/text-to-amr", params=params)
        r = r.json()
        return r['penman']
