#!/usr/bin/evn python2.7
# -*- coding: utf-8 -*-

"""
IO模块
数据加载与保存
"""
import os
import sys
import json
import pandas as pd
from logger import logger


def dump_urls(urls, file_path, csv=True):
    """
    :param urls:
    :param file_path:
    :param csv:
    :return:
    """
    if os.path.isfile(file_path):
        os.remove(file_path)
        logger.debug("OLD DATA FIND! REMOVING\t%s" % file_path)
    try:
        if csv:
            df = pd.DataFrame({"url": urls})
            df.to_csv(file_path, index=False)
        else:
            with open(file_path, "w") as fd:
                for line in urls:
                    fd.write(line + "\n")
        logger.debug("URLs has been dump\t%s" % file_path)
    except Exception as e:
        logger.error("%s\tFILE DUMP ERROR %s" % (file_path, str(e)))
        sys.exit(0)


def load_urls(file_path, csv=True):
    """
    :param file_path:
    :param csv:
    :return:
    """
    try:
        if csv:
            df = pd.read_csv(file_path)
            urls = list(df.url)
        else:
            with open(file_path, "r") as fd:
                urls = [_.strip() for _ in fd]
        logger.debug("URLs Count:\t%d" % (len(urls)))
        return urls
    except Exception as e:
        logger.error("%s FILE OPEN ERROR! %s" % (file_path, str(e)))
        sys.exit(0)


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
        logger.error("%s\tFILE DUMP ERROR! %s" % (file_path, str(e)))
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
        logger.error("%s\t FILE OPEN ERROR! %s" % (file_path, str(e)))
        sys.exit(0)
    return df


# init dump cluster data
# 由于字符串相似度聚类需要较长时间，将聚好的类顺序写入文件
def _init_dump_cluster_data(file_path):
    """
    json format
    :param file_path: cluster file path [str]
    :return:
    """
    if os.path.isfile(file_path):
        os.remove(file_path)
        logger.debug("OLD DATA FIND! REMOVING\t%s" % file_path)


# dump cluster data
# 由于字符串相似度聚类需要较长时间，将聚好的类顺序写入文件
def _dump_cluster_data(file_path, single_cluster, index):
    """
    json format
    :param file_path: cluster file path [str]
    :param single_cluster: single cluster [list]
    :param index: cluster index [int]
    :return:
    """
    with open(file_path, "a+") as fd:
        fd.write(json.dumps({index: single_cluster}) + '\n')


# load k means / hierarchical cluster data
def _load_kmeans_hier_cluster_data(file_path):
    """
    json format
    :param file_path: file_path [str]
    :return: list of cluster [list]
    """
    res = []
    try:
        with open(file_path, 'r') as fd:
            res_dict = json.loads(fd.read().strip())
        for i in range(len(res_dict)):
            res.append(res_dict[str(i)])
        logger.debug("K-means/hier data has been loaded\t%s" % file_path)
    except Exception as e:
        logger.error("%s\t FILE OPEN ERROR!\t%s" % (file_path, str(e)))
        sys.exit(0)
    return res


# dump k mean / hierarchical cluster data
def _dump_kmeans_hier_cluster_data(cluster_dict, file_path):
    """
    json format
    :param cluster_dict: cluster results [dict]
    :param file_path:
    :return:
    """
    if os.path.isfile(file_path):
        os.remove(file_path)
        logger.debug("OLD DATA FIND! REMOVING\t%s" % file_path)
    try:
        with open(file_path, 'w+') as fd:
            fd.write(json.dumps(cluster_dict))
        logger.debug("cluster has been dump\t%s" % file_path)
    except Exception as e:
        logger.error("%s\tFILE DUMP ERROR! %s" % (file_path, str(e)))
        sys.exit(0)


# load  string distance cluster data
def _load_cluster_data(file_path):
    """
    json format
    :param file_path:
    :return: urls list [list]
    """
    cluster = []
    try:
        with open(file_path, 'r') as fd:
            for line in fd:
                cluster.append(json.loads(line.strip()))
        logger.debug("Cluster Data has been loaded\t%s" % file_path)
    except Exception as e:
        logger.error("%s\t FILE OPEN ERROR! %s" % (file_path, str(e)))
        sys.exit(0)
    cluster = [_[_.keys()[0]] for _ in cluster]
    return cluster


# load regular expression data
def _load_regex_list(file_path):
    """
    :param file_path:
    :return:
    """
    regex = []
    try:
        with open(file_path, 'r') as fd:
            for line in fd:
                regex.append(line.strip())
        logger.debug("Regex Data has been loaded\t%s" % file_path)
    except Exception as e:
        logger.error("%s\t FILE OPEN ERROR! %s" % (file_path, str(e)))
        sys.exit(0)
    return regex


# load regular expression data
def _dump_regex_list(regex_list, file_path):
    """
    :param file_path:
    :param regex_list: [list]
    :return:
    """
    if os.path.isfile(file_path):
        os.remove(file_path)
        logger.debug("OLD DATA FIND! REMOVING\t%s" % file_path)
    try:
        with open(file_path, 'w+') as fd:
            for regex in regex_list:
                fd.write(regex + '\n')
        logger.debug("Regex has been dump\t%s" % file_path)
    except Exception as e:
        logger.error("%s\tFILE DUMP ERROR %s" % (file_path, str(e)))
        sys.exit(0)


# load test data in csv url format
def _load_test_data(file_path, csv=True):
    """
    :param file_path:
    :return: urls [list]
    """
    try:
        if csv:
            df = pd.read_csv(file_path)
            urls = list(df.url)
        else:
            with open(file_path, "r") as f:
                urls = [_.strip() for _ in f]
        logger.debug("Test Data has been loaded\t%s" % file_path)
    except Exception as e:
        logger.error("%s\t FILE OPEN ERROR! %s" % (file_path, str(e)))
        sys.exit(0)
    return urls


# dump check result
def _dump_check_result(fp, tp, regex_list, file_path):
    """
    :param fp: [list]
    :param tp: [list]
    :param regex_list: [list]
    :param file_path:  [str]
    :return:
    """
    if os.path.isfile(file_path):
        os.remove(file_path)
        logger.debug("OLD DATA FIND! REMOVING\t%s" % file_path)
    try:
        with open(file_path, "w+") as fd:
            for i in range(len(regex_list)):
                fd.write(str(fp[i]) + "\t" + str(tp[i]) +
                         "\t" + regex_list[i] + "\n")
        logger.debug("Check Result has been dump\t%s" % file_path)
    except Exception as e:
        logger.error("%s\tFILE DUMP ERROR %s" % (file_path, str(e)))
        sys.exit(0)


# load check result
def _load_check_result(file_path):
    """
    :param file_path:
    :return:
    """
    try:
        fp_list, tp_list, regex_list = [], [], []
        with open(file_path, "r") as fd:
            for line in fd:
                res = line.strip().split("\t")
                fp_list.append(int(res[0]))
                tp_list.append(int(res[1]))
                regex_list.append(res[2])
        df = pd.DataFrame({"fp": fp_list, "tp": tp_list, "regex": regex_list})
        logger.debug("check Data has been loaded\t%s" % file_path)
    except Exception as e:
        logger.error("%s\t FILE OPEN ERROR! %s" % (file_path, str(e)))
        sys.exit(0)
    return df
