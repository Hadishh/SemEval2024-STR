#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hadi Sheikhi'
__email__ = 'hsheikhi@ualberta.ca'


# dependency
# built-in
import re
from sentence_transformers import SentenceTransformer, util
from src.utils.translator import translate_texts
from tqdm import tqdm

class Model(object):
    """
    adapted from https://github.com/semantic-textual-relatedness/Semantic_Relatedness_SemEval2024/blob/main/STR_Baseline.ipynb
    """
    def __init__(self, model_name, src_lang, tgt_lang, batch_size=32):
        super(Model, self).__init__()
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.similarity_method = "cosine"
        self.source_language = src_lang
        self.target_langauge = tgt_lang
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
    
    def distance(self, e1, e2):
        match self.similarity_method:
            case "cosine":
                cosine_distance = (1 - util.cos_sim(e1, e2).item()) / 2
                return round(1 - cosine_distance, 2)


    def predict(self, s1s, s2s):
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
        
        embeddings1 = self.model.encode(s1s, batch_size=self.batch_size)
        embeddings2 = self.model.encode(s2s, batch_size=self.batch_size)
        
        return [self.distance(s1, s2) for (s1, s2) in zip(s1s, s2s)]