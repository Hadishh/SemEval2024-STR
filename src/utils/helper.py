#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Author'
__email__ = 'Email'


# dependency
# built-in
import sys
import logging
# private
from src.methods import base, sentence_transformers, amr, sentence_transformers_translation, amr_translation

def str2bool(v):
    """Method to map string to bool for argument parser"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def init_logger(config):
    """initialize the logger"""
    file_handler = logging.FileHandler(filename=config.LOG_TXT)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        encoding='utf-8'
        , format='%(asctime)s | %(message)s'
        , datefmt='%Y-%m-%d %H:%M:%S'
        , level=logging.INFO
        , handlers=handlers
        )
    logger = logging.getLogger(__name__)
    return logger

def get_model(config):
    match config.method:
        case 'base':
            return base.Model()
        case 'sentence-transformers':
            return sentence_transformers.Model(config.model_name)
        case 'amr':
            return amr.AMR(config)
        case "sentence-transformers-translation":
            return sentence_transformers_translation.Model(config.model_name, config.tgt_lan, config.translate_lang)
        case 'amr-translation':
            return amr_translation.AMR(config)
        case _:
            raise NotImplementedError