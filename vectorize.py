#!/usr/bin/evn python2.7
# -*- coding: utf-8 -*-

"""
向量化URL
"""
import numpy as np
import pandas as pd
from logger import logger

from config import cfg
import urlnormalize
from io import _dump_vector_data

ASCII_SIZE = cfg.ASCII_SIZE


# vectorize
def _core_vectorize(url):
    assert isinstance(url, str)
    res = np.zeros((1, ASCII_SIZE), dtype=np.int32)
    try:
        for char in url:
            # 不考虑超过128的ascii的特殊字符
            if ord(char) < 128:
                res[0][ord(char)] += 1
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
