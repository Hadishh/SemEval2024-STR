#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hadi Sheikhi'
__email__ = 'hsheikhi@ualberta.ca'


# dependency
# built-in
import re
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import torch
from torch import Tensor
class Model(object):
    """
    adapted from https://github.com/semantic-textual-relatedness/Semantic_Relatedness_SemEval2024/blob/main/STR_Baseline.ipynb
    """
    def __init__(self, model_name, batch_size=32):
        super(Model, self).__init__()
        # self.model = AutoModel.from_pretrained(model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = SentenceTransformer(model_name)

        self.batch_size = batch_size
        self.similarity_method = "cosine"
    
    def distance(self, e1, e2):
        match self.similarity_method:
            case "cosine":
                cosine_distance = (1 - util.cos_sim(e1, e2).item()) / 2
                return round(1 - cosine_distance, 2)
    
    def predict(self, s1s, s2s):
        
        embeddings1 = self.model.encode(s1s, batch_size=self.batch_size)
        embeddings2 = self.model.encode(s2s, batch_size=self.batch_size)
        
        return [self.distance(e1, e2) for e1, e2 in zip(embeddings1, embeddings2)], None