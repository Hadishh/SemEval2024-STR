#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hadi Sheikhi'
__email__ = 'hsheikhi@ualberta.ca'


# dependency
# built-in
import os, random
# public
import numpy as np
import pandas as pd
# private
from config import Config
from src.utils import helper


class SR(object):
    """
    docstring for Semantic Relatedness SemEval2024
    """
    def __init__(self):
        super(SR, self).__init__()
        self.initialize()

    def initialize(self):
        # init configurations
        self.config = Config()
        # set random seed
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        # get logger
        self.logger = helper.init_logger(self.config)
        self.logger.info('Logger initialized.')
        self.logger.info('='*5 + 'Configurations' + '='*5)
        for k, v in self.config.__dict__.items():
            self.logger.info(f'{k}: {v}')
        # get model
        self.model = helper.get_model(self.config)

    def dev(self):
        self.logger.info('='*5 + 'Dev' + '='*5)
        # read in dev datasets
        raw_df = pd.read_csv(self.config.DEV_CSV, encoding="utf-8")
        # xs1, xs2 = map(list, zip(*[tuple(row['Text'].split('\t')) for idx, row in raw_df.iterrows()]))
        xs1, xs2 = map(list, zip(*[(row['Text1 Translation'], row['Text2 Translation']) for idx, row in raw_df.iterrows()]))
        self.logger.info('Data loaded.')
        # model to predict
        ys_, s1s, s2s = self.model.predict(xs1, xs2)
        # Submission file has two columns: 'PairID' and 'Pred_Score'
        # raw_df['Pred_Score'] = ys_
        raw_df['Text1'] = xs1
        raw_df["Text2"] = xs2
        raw_df["Text1 Translation"] = s1s
        raw_df["Text2 Translation"] = s2s
        raw_df[['PairID', 'Text1', "Text1 Translation", "Text2", "Text2 Translation"]].to_csv(
            "mar2eng_train.csv"
            , index=False,
            encoding="utf-8"
            )
        self.logger.info('Results saved as {}.'.format(self.config.RESULTS_CSV))
        self.logger.info('Done.')
    
    def test(self):
        self.logger.info('='*5 + 'TEST' + '='*5)
        # read in dev datasets
        raw_df = pd.read_csv(self.config.TEST_CSV, encoding="utf-8")
        # xs1, xs2 = map(list, zip(*[tuple(row['Text'].split('\t')) for idx, row in raw_df.iterrows()]))
        xs1, xs2 = map(list, zip(*[(row['Text1 Translation'], row['Text2 Translation']) for idx, row in raw_df.iterrows()]))
        self.logger.info('Data loaded.')
        # model to predict
        ys_= self.model.predict(xs1, xs2)
        # Submission file has two columns: 'PairID' and 'Pred_Score'
        raw_df['Pred_Score'] = ys_
        raw_df[['PairID', 'Pred_Score']].to_csv(
            self.config.RESULTS_CSV,
            index=False,
            encoding="utf-8"
            )
        self.logger.info('Results saved as {}.'.format(self.config.RESULTS_CSV))
        self.logger.info('Done.')
    
def main():
    sr = SR()
    sr.dev()

if __name__ == '__main__':
      main()
