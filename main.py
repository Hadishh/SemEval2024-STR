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
    
    def __inference(self, input_csv, output_csv, split):
        raw_df = pd.read_csv(input_csv, encoding="utf-8")
        if self.config.tgt_lan == "eng":
            xs1, xs2 = map(list, zip(*[tuple(row['Text'].split('\n')) for idx, row in raw_df.iterrows()]))
        else:
            xs1, xs2 = map(list, zip(*[(row['Text1 Translation'], row['Text2 Translation']) for idx, row in raw_df.iterrows()]))
        
        self.logger.info('Data loaded.')
        # model to predict
        ys_, metadata = self.model.predict(xs1, xs2)
        df = pd.DataFrame(metadata)
        df.to_csv(os.path.join(self.config.RESULTS_PATH, f"{split}_metadata.csv"), index=None)
        # Submission file has two columns: 'PairID' and 'Pred_Score'
        raw_df['Pred_Score'] = ys_
        raw_df[['PairID', 'Pred_Score']].to_csv(
            output_csv,
            index=False,
            encoding="utf-8"
            )
        self.logger.info('Results saved as {}.'.format(output_csv))
        self.logger.info('Done.')
    
    def dev(self):
        self.logger.info('='*5 + 'Dev' + '='*5)
        self.__inference(self.config.DEV_CSV, self.config.RESULTS_DEV_CSV, "dev")
    
    def test(self):
        self.logger.info('='*5 + 'TEST' + '='*5)
        self.__inference(self.config.TEST_CSV, self.config.RESULTS_TEST_CSV, "test")
    
def main():
    sr = SR()
    sr.dev()
    sr.test()

if __name__ == '__main__':
      main()
