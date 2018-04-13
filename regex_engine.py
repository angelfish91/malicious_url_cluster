#!/usr/bin/evn python2.7
# -*- coding: utf-8 -*-

"""
正则表达式提取模块，包含如下功能
1. 中间结果的输入输出
2. 正则表达式抽取
3. 正则表达式验证
"""

import os
import sys
import re
import json
import time
import random
import pandas as pd
from collections import Counter
from joblib import Parallel, delayed


from urlnormalize import UrlNormalize
from logger import logger
from config import cfg
from lib.max_substring import maxsubstring


N_JOBS = cfg.GLOBAL_N_JOBS

# regular expression extract
MIN_SUBSTRING_SIZE = cfg.MIN_SUBSTRING_SIZE
SINGLE_REGEX_SIZE = cfg.SINGLE_REGEX_SIZE
TOTAL_REGEX_SIZE = cfg.TOTAL_REGEX_SIZE
SINGLE_REGEX_SIZE_RATIO = cfg.SINGLE_REGEX_SIZE_RATIO
TOTAL_REGEX_SIZE_RATIO = cfg.TOTAL_REGEX_SIZE_RATIO

# regular expression extract
BIG_CLUSTER_SAMPLE_ROUND = cfg.BIG_CLUSTER_SAMPLE_ROUND
SMALL_CLUSTER_SIZE = cfg.SMALL_CLUSTER_SIZE
BIG_CLUSTER_SIZE = cfg.BIG_CLUSTER_SIZE

# regular expression  publish
PUBLISH_FP_THRESH = cfg.PUBLISH_FP_THRESH
PUBLISH_TP_THRESH = cfg.PUBLISH_TP_THRESH
PUBLISH_RATIO = cfg.PUBLISH_RATIO
PUBLISH_RATIO_TP_THRESH = cfg.PUBLISH_RATIO_TP_THRESH

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


# convert string to regular expression
def _regularize_string(string):
    """
    transfer_mean = "$()*+.[]?\^{},|"
    :param string:
    :return:
    """
    string = string.replace("$", "\$")
    string = string.replace("(", "\(")
    string = string.replace(")", "\)")
    string = string.replace("*", "\*")
    string = string.replace("+", "\+")
    string = string.replace(".", "\.")
    string = string.replace("[", "\[")
    string = string.replace("]", "\]")
    string = string.replace("?", "\?")
    string = string.replace("^", "\^")
    string = string.replace("{", "\{")
    string = string.replace("}", "\}")
    string = string.replace(",", "\,")
    string = string.replace("|", "\|")
    return string


# regular expression match
def _regex_match(regex, url):
    """
    :param regex: input regular expression
    :param url: input url
    :return: whether the expression match the url
    """
    pattern = re.compile(regex)
    res = pattern.search(url)
    if res is None:
        return False
    else:
        return True


# sorting the output n max long string
def _sorting_regex(string, regex):
    """
    :param string: string in clusters
    :param regex: output n max long string
    :return: sorted regular expression
    """
    indices = [string.find(_) for _ in regex]
    pair_indices_regex = [(indices[i], regex[i]) for i in range(len(regex))]
    pair_indices_regex = sorted(pair_indices_regex, key=lambda x: x[0])
    return [_[1] for _ in pair_indices_regex]


# search regex in clusters
def _regex_search_engine(cluster):
    """
    :param cluster: url cluster
    :return: None or regex
    """
    assert len(cluster) >= 2
    # 由于公共子串搜索性能的问题，当字符串长度超出一定长度阈值时，重新进行采样
    if len(cluster[0]) > 512:
        cluster = cluster[:2]
        cluster = [_[:512] for _ in cluster]
    if len(cluster[0]) > 256:
        cluster = cluster[:3]

    # search part 1) max substring 2) sorting 3) size check 4) regularize
    # 1)
    regex = maxsubstring(cluster, thresh=MIN_SUBSTRING_SIZE)
    # 2)
    regex = _sorting_regex(cluster[0], regex)
    # 3)
    if len(regex) == 0:
        return None
    if len(regex) == 1 and len(
        regex[0]) < SINGLE_REGEX_SIZE_RATIO * len(
        cluster[0]) or len(
            regex[0]) < SINGLE_REGEX_SIZE:
        return None
    if len(regex) > 1:
        total_size = sum([len(_) for _ in regex])
        if total_size < TOTAL_REGEX_SIZE_RATIO * len(cluster[0]) or \
                total_size < TOTAL_REGEX_SIZE:
            return None
    # 4)
    regex = [_regularize_string(_) for _ in regex]
    regex = "(.)+".join(regex)
    return regex


# search regular expression in big cluster
def _regex_search_in_big_cluster(cluster):
    """
    :param cluster:  url string cluster [list]
    :return: regular expression match most of the cluster
    """
    assert len(cluster) >= 2
    # sampling the big cluster and maximize match count
    random_sample_list = []
    match_count_list = []
    regex_list = []
    # prepare the sample list
    for i in range(BIG_CLUSTER_SAMPLE_ROUND):
        random_sample_list.append(
            random.sample(cluster, int(len(cluster) / 2)))
    # get sample regex
    sample_regex_list = Parallel(
        n_jobs=N_JOBS)(
        delayed(_regex_search_engine)(
            random_sample)
        for random_sample in random_sample_list)
    # analysis the above regex
    for sample_regex in sample_regex_list:
        if sample_regex is None:
            continue
        match_count_list.append(
            sum([_regex_match(sample_regex, _) for _ in cluster]))
        regex_list.append(sample_regex)
    if len(match_count_list) == 0:
        return None
    max_match_count_index = match_count_list.index(max(match_count_list))
    # log big cluster on console for optimize
    if max_match_count_index != 0:
        for i in cluster:
            logger.debug(str(i))
        logger.debug(
            "index:\t%d\t%s" %
            (max_match_count_index, str(
                regex_list[max_match_count_index])))
    return regex_list[max_match_count_index]


# search regular expression in small cluster
def _regex_search_in_small_cluster(cluster):
    """
    :param cluster: url string cluster
    :return: regular expression match all the url string
    """
    assert len(cluster) >= 2
    regex = _regex_search_engine(cluster)
    return regex


# extract regular expression form small clusters
def regex_extract(input_file_path,
                  output_file_path):
    """
    :param input_file_path: cluster file path
    :param output_file_path: regular expression file path
    :return: None
    """
    start_time = time.time()
    cluster_list = _load_cluster_data(input_file_path)
    cluster_size = [len(_) for _ in cluster_list]
    # log detail info of cluster on console
    logger.debug("total cluster num:\t%d" % len(cluster_list))
    logger.debug("big cluster:\t%d" %
                 len([1 for i in cluster_size if i >= BIG_CLUSTER_SIZE]))
    logger.debug("small cluster:\t%d" % len(
        [1 for i in cluster_size if SMALL_CLUSTER_SIZE <= i < BIG_CLUSTER_SIZE]))
    logger.debug("single one:\t%d" % len([1 for i in cluster_size if i == 1]))
    logger.debug("cluster size detail:\t%s" % str(Counter(cluster_size)))
    # treat different for different size of cluster
    regex_list = []
    for cluster_index, cluster in enumerate(cluster_list):

        if SMALL_CLUSTER_SIZE <= len(cluster) < BIG_CLUSTER_SIZE:
            regex = _regex_search_in_small_cluster(cluster)
            regex_list.append(regex)
        if len(cluster) >= BIG_CLUSTER_SIZE:
            regex = _regex_search_in_big_cluster(cluster)
            regex_list.append(regex)

    regex_list = [_ for _ in regex_list if _ is not None]
    regex_list = list(set(regex_list))
    _dump_regex_list(regex_list, output_file_path)

    logger.debug("extract regex count:\t%d" % len(regex_list))
    logger.debug("extract regex time cost:\t%f" % (time.time() - start_time))


# check the extracted regular expression with white list url to avoid FP
def _check_performance(
        regex_list,
        benign_urls,
        malicious_urls,
        batch_index,
        n_jobs):
    """
    split regular expression into batches for multi-process check
    :param regex_list: regular expressions to check [list]
    :param benign_urls: white list url [list]
    :param malicious_urls: black list url [list]
    :param batch_index: batch index
    :param n_jobs: n jobs for multi-process
    :return: (FP, TP) [list]
    """
    # decides which batch of regular expression use to check
    batch_size = int(len(regex_list) / n_jobs)
    start_index = batch_index * batch_size
    end_index = (batch_index + 1) * batch_size
    if batch_index == n_jobs - 1:
        end_index += n_jobs
    res = []
    # check batch regular expression with white list urls
    for index, regex in enumerate(regex_list[start_index: end_index]):
        benign_res = sum([_regex_match(regex, _) for _ in benign_urls])
        malicious_res = sum([_regex_match(regex, _) for _ in malicious_urls])
        print(
            "batch index",
            batch_index,
            "sample index",
            index,
            "FP",
            benign_res,
            "TP",
            malicious_res)
        res.append((benign_res, malicious_res))
    return res


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


# regular expression check and publish
def regex_check(input_file_path,
                test_benign_file_path,
                test_malicious_file_path,
                result_file_path,
                n_jobs=N_JOBS):

    regex = _load_regex_list(input_file_path)
    benign_urls = _load_test_data(test_benign_file_path)
    malicious_urls = _load_test_data(test_malicious_file_path)

    temp_res = Parallel(
        n_jobs=n_jobs)(
        delayed(_check_performance)(
            regex,
            benign_urls,
            malicious_urls,
            batch_index,
            n_jobs) for batch_index in range(n_jobs))

    res = []
    for i in temp_res:
        res = res + i
    assert len(res) == len(regex)
    res_fp = [_[0] for _ in res]
    res_tp = [_[1] for _ in res]

    # dump regular expression result
    _dump_check_result(res_fp, res_tp, regex, result_file_path)


def regex_publish(result_file_path,
                  publish_file_path,
                  publish_fp_thresh=PUBLISH_FP_THRESH,
                  publish_tp_thresh=PUBLISH_TP_THRESH,
                  publish_ratio=PUBLISH_RATIO,
                  publish_ratio_tp_thresh=PUBLISH_RATIO_TP_THRESH):
    df = _load_check_result(result_file_path)
    # dump regular expression with 0 fp
    dfa = df.loc[df.fp <= publish_fp_thresh]
    dfa = dfa.loc[df.tp >= publish_tp_thresh]
    regex_list_publish = list(dfa.regex)

    df["ratio"] = (df.fp.values + 1) / (df.tp.values + 1)
    dfb = df.loc[df.ratio < publish_ratio]
    dfb = dfb.loc[dfb.fp <= publish_ratio_tp_thresh]
    regex_list_publish += list(dfb.regex)
    _dump_regex_list(regex_list_publish, publish_file_path)


# evaluate the regex extracted from the list
def regex_evaluate(publish_file_path,
                   test_malicious_file_path):
    malicious_urls = _load_test_data(test_malicious_file_path)
    regex_list = _load_regex_list(publish_file_path)
    malicious_urls_hit = set()
    for index, regex in enumerate(regex_list):
        for url in malicious_urls:
            if _regex_match(regex, url):
                malicious_urls_hit.add(url)
    hit_percentage = float(len(malicious_urls_hit)) / \
        float(len(set(malicious_urls))) * 100
    logger.debug("hit percentage: %f" % hit_percentage)


def _core_predict(regex_list, test_urls, batch_index, n_jobs):
    """
    split regular expression into batches for multi-process check
    :param regex_list: regular expressions to check [list]
    :param test_urls: white list url [list]
    :param batch_index: batch index
    :param n_jobs: n jobs for multi-process
    :return:
    """
    # decides which batch of regular expression use to check
    batch_size = int(len(regex_list) / n_jobs)
    start_index = batch_index * batch_size
    end_index = (batch_index + 1) * batch_size
    if batch_index == n_jobs - 1:
        end_index += n_jobs

    # check batch regular expression with white list urls
    res = dict()
    for index, i in enumerate(regex_list[start_index: end_index]):
        hit = []
        for url in test_urls:
            worker = UrlNormalize(url)
            url_plus = worker.get_quote_plus_url()
            if _regex_match(i, url_plus):
                hit.append(url)
        print(
            "batch index",
            batch_index,
            "sample index",
            index,
            "hit", len(hit)
        )
        if len(hit) != 0:
            res["\t".join(i)] = hit
    return res


# predict for unknown urls
def malicious_url_predict(input_file_path,
                          regex_file_path,
                          n_jobs=N_JOBS):
    """
    :param input_file_path:
    :param regex_file_path:
    :param n_jobs:
    :return: predict_malicious [list], predict_dict [dict]
    """
    regex = _load_regex_list(regex_file_path)
    test_url = _load_test_data(input_file_path)
    # predict part
    predict_res = Parallel(
        n_jobs=n_jobs)(
        delayed(_core_predict)(
            regex,
            test_url,
            index,
            n_jobs) for index in range(n_jobs))
    # precess the result
    predict_malicious = []
    predict_dict = dict()
    for i in predict_res:
        for j in i:
            predict_malicious.extend(i[j])
            predict_dict[j] = i[j]
    return predict_malicious, predict_dict
