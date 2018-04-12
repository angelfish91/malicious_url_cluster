#!/usr/bin/evn python2.7
# -*- coding: utf-8 -*-

"""
向量化URL
"""
import os
import sys
import numpy as np
import pandas as pd
from logger import logger

from config import cfg
import urlnormalize


ASCII_SIZE = cfg.ASCII_SIZE


def _dump_vector_data(file_path, df_vector):
    if os.path.isfile(file_path):
        os.remove(file_path)
        logger.debug("OLD DATA FIND! REMOVING\t%s" % file_path)
    try:
        df_vector.to_csv(file_path)
        logger.debug("vector has beeen dump\t%s" % file_path)
    except Exception as e:
        logger.warning("%s\tFILE DUMP ERROR! %s" % (file_path, e))
        sys.exit(0)

def _load_vector_data(file_path):
    try:
        df = pd.read_csv(file_path, index_col='url')
        logger.debug("vector data has been loaded\t%s" % file_path)
    except Exception as e:
        logger.warning("%s\t FILE OPEN ERROR! %s" % (file_path, e))
        sys.exit(0)
    return df

def _vectorize(url):
    assert isinstance(url, str)
    res = np.zeros((1, ASCII_SIZE), dtype=np.int32)
    try:
        for char in url:
            # 不考虑超过128的ascii的特殊字符
            try:
                res[0][ord(char)] += 1
            except Exception as e:
                pass
    except Exception as e:
        logger.error("VECTORIZED ERROR\t%s\t%s" % (url, str(e)))
    return res


def make_vectorize(
        urls,
        domain=True,
        path=False,
        param=False,
        output_path=cfg.VECTOR_DOMAIN_DATA,
        dump=True):
    """
    :param urls:
    :param domain: whether use domain to build vector
    :param path: whether use path to build vector
    :param param: whether use param to build vector
    :param output_path:
    :return:
    """
    assert isinstance(urls, list)

    urls_domain, urls_path, urls_params = [], [], []
    urls_processed = []
    for i in urls:
        worker = urlnormalize.UrlNormalize(i)
        if domain:
            urls_domain.append(worker.get_hostname())
        if path:
            urls_path.append(worker.get_path())
        if param:
            urls_params.append(worker.get_params())

    if domain:
        urls_processed.append(locals()["urls_domain"])
    if path:
        urls_processed.append(locals()["urls_path"])
    if param:
        urls_processed.append(locals()["urls_params"])

    res = []
    for i in range(len(urls_processed)):
        temp_res = [_vectorize(_) for _ in urls_processed[i]]
        temp_res = np.concatenate(temp_res, axis=0)
        res.append(temp_res)

    res = np.concatenate(res, axis=1)
    df = pd.DataFrame(res)
    df['url'] = urls
    df = df.set_index('url')

    logger.debug("vectorization complete! data shape:\t%s" %
                 str(df.values.shape))
    if dump:
        _dump_vector_data(output_path, df)
    else:
        return df
