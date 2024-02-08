import requests
from src.smatch.smatch import get_amr_match, compute_f
from tqdm import tqdm

class AMR(object):
    """
    adapted from https://github.com/semantic-textual-relatedness/Semantic_Relatedness_SemEval2024/blob/main/STR_Baseline.ipynb
    """
    def __init__(self):
        super(AMR, self).__init__()

    def smatch_score(self, s1, s2):
        amr1 = self.__get_amr(s1)
        amr2 = self.__get_amr(s2)
        scores = get_amr_match(amr1, amr2)
        p, r, f = compute_f(scores[0], scores[1], scores[2])
        return f

    def predict(self, s1s, s2s):
        scores = []
        data = []
        for s1, s2 in tqdm(zip(s1s, s2s)):
            scores.append(self.smatch_score(s1, s2))
        return scores
    
    def __get_amr(self, s):
        params= {"sentence": s}
        r = requests.get("https://nlp.uniroma1.it/spring/api/text-to-amr", params=params)
        r = r.json()
        return r['penman']
