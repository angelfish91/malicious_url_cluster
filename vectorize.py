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


# dump vector data
def _dump_vector_data(file_path, df_vector):
    """
    :param file_path: file_path [str]
    :param df_vector: vector [pd.DataFrame]
    :return:
    """
    if os.path.isfile(file_path):
        os.remove(file_path)
        logger.debug("OLD DATA FIND! REMOVING\t%s" % file_path)
    try:
        df_vector.to_csv(file_path)
        logger.debug("vector has beeen dump\t%s" % file_path)
    except Exception as e:
        logger.error("%s\tFILE DUMP ERROR! %s" % (file_path, e))
        sys.exit(0)


# load vector data
def _load_vector_data(file_path):
    """
    load vector data in csv format
    :param file_path: file_path [str]
    :return:
    """
    try:
        df = pd.read_csv(file_path, index_col='url')
        logger.debug("vector data has been loaded\t%s" % file_path)
    except Exception as e:
        logger.error("%s\t FILE OPEN ERROR! %s" % (file_path, e))
        sys.exit(0)
    return df


# vectorize
def _core_vectorize(url):
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


# make url vector
def make_vectorize(
        urls,
        domain=True,
        path=False,
        param=False,
        output_path=None,
        dump=True):
    """
    :param urls: list of urls [list]
    :param domain: whether use domain to build vector [bool]
    :param path: whether use path to build vector [bool]
    :param param: whether use param to build vector [bool]
    :param output_path: [str]
    :param dump: whether dump vector [bool]
    :return: pandas DataFrame [pd.DataFrame]
    """
    assert isinstance(urls, list)

    urls_domain, urls_path, urls_params = [], [], []
    urls_processed = []
    for url in urls:
        worker = urlnormalize.UrlNormalize(url)
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
    for index in range(len(urls_processed)):
        temp_res = [_core_vectorize(_) for _ in urls_processed[index]]
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
