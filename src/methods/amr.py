import requests
from src.smatch.smatch import get_amr_match, compute_f
from tqdm import tqdm
import pandas as pd
import os

class AMR(object):
    """
    adapted from https://github.com/semantic-textual-relatedness/Semantic_Relatedness_SemEval2024/blob/main/STR_Baseline.ipynb
    """
    def __init__(self, config):
        super(AMR, self).__init__()
        self.config = config

    def smatch_score(self, s1, s2):
        amr1 = self.__get_amr(s1)
        amr2 = self.__get_amr(s2)
        scores = get_amr_match(amr1, amr2)
        p, r, f = compute_f(scores[0], scores[1], scores[2])
        return f, amr1, amr2

    def predict(self, s1s, s2s):
        scores = []
        data = []
        for s1, s2 in zip(tqdm(s1s), s2s):
            score, amr1, amr2 = self.smatch_score(s1, s2)
            scores.append(score)
            data.append({"s1": s1, "amr1": amr1, "s2": s2, "amr2": amr2, "score": score})  
        return scores, data
    
    def __get_amr(self, s):
        params= {"sentence": s}
        r = requests.get("https://nlp.uniroma1.it/spring/api/text-to-amr", params=params)
        r = r.json()
        return r['penman']
