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
import time
import random
import pandas as pd
from collections import Counter, defaultdict
from joblib import Parallel, delayed


from urlnormalize import UrlNormalize
from logger import logger
from config import cfg
from lib.max_substring import maxsubstring
from file_io import _load_cluster_data, _load_regex_list, _dump_regex_list, _load_test_data, \
    _dump_check_result, _load_check_result

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
def _url_regex_match(regex, url):
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


# replace contine num in hostname
def _continue_num_regex(url, regex):

    # port subsitude
    regex_list = regex.split("/")
    netloc = regex_list[0]
    pattern = re.compile(netloc)
    if pattern.search(url.split("/")[0]) is None:
        return regex
    port = r":[0-9]{2,}"
    port_f = re.compile(port).search(netloc)
    if port_f is not None:
        netloc = netloc.replace(
            netloc[port_f.span()[0]:port_f.span()[1]], port)
        regex_list[0] = netloc
    return "/".join(regex_list)


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

    # 1) search for max substring
    regex = maxsubstring(cluster, thresh=MIN_SUBSTRING_SIZE)
    # 2) size check
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
    # 3) regularize
    regex = [_regularize_string(_) for _ in regex]
    regex = "(.)+".join(regex)
    # 4) sub continue digit in hostname
    regex = _continue_num_regex(cluster[0], regex)
    return regex


def _regex_search_in_big_cluster_opt(cluster):
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
        random_sample = random.sample(cluster, int(len(cluster) / 2))
        sample_regex = _regex_search_engine(random_sample)
        if sample_regex is None:
            continue
        match_count_list.append(
            sum([_url_regex_match(sample_regex, _) for _ in cluster]))
        regex_list.append(sample_regex)
    if len(match_count_list) == 0:
        return None
    max_match_count_index = match_count_list.index(max(match_count_list))
    return regex_list[max_match_count_index]    


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
            sum([_url_regex_match(sample_regex, _) for _ in cluster]))
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
def url_regex_extract(input_file_path,
                  output_file_path, dump=True):
    """
    :param input_file_path: cluster file path
    :param output_file_path: regular expression file path
    :param dump: whether dump
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
        if cluster_index%100 == 0:
            logger.debug("step\t%d" % cluster_index)

    regex_list = [_ for _ in regex_list if _ is not None]
    regex_list = list(set(regex_list))
    logger.debug("extract regex count:\t%d" % len(regex_list))
    logger.debug("extract regex time cost:\t%f" % (time.time() - start_time))

    if dump:
        _dump_regex_list(regex_list, output_file_path)
    else:
        return regex_list
    

    
def _core_url_regex_extract_opt(cluster_index, cluster):
    if SMALL_CLUSTER_SIZE <= len(cluster) < BIG_CLUSTER_SIZE:
        return _regex_search_in_small_cluster(cluster)
    if len(cluster) >= BIG_CLUSTER_SIZE:
        return _regex_search_in_big_cluster_opt(cluster)
    if cluster_index % 200 == 0:
        print "step: %d" %cluster_index
    return None
            
    
# extract regular expression form small clusters
def url_regex_extract_opt(input_file_path,
                 output_file_path, dump=True):
    """
    :param input_file_path: cluster file path
    :param output_file_path: regular expression file path
    :param dump: whether dump
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
    regex_list = Parallel(n_jobs=N_JOBS)(
        delayed(_core_url_regex_extract_opt)(
            cluster_index, cluster)
        for cluster_index, cluster in enumerate(cluster_list))
    
    regex_list = [_ for _ in regex_list if _ is not None]
    regex_list = list(set(regex_list))
    logger.debug("extract regex count:\t%d" % len(regex_list))
    logger.debug("extract regex time cost:\t%f" % (time.time() - start_time))

    if dump:
        _dump_regex_list(regex_list, output_file_path)
    else:
        return regex_list


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
        benign_res = sum([_url_regex_match(regex, _) for _ in benign_urls])
        malicious_res = sum([_url_regex_match(regex, _)
                             for _ in malicious_urls])
        print "batch index %d\tsample index %d\tFP %d\tTP %d\t%s" \
              %(batch_index, index, benign_res, malicious_res, regex)
        res.append((benign_res, malicious_res))
    return res


# regular expression check and publish
def url_regex_check(input_file_path,
                    test_benign_file_path,
                    test_malicious_file_path,
                    result_file_path,
                    n_jobs=N_JOBS):
    """
    :param input_file_path:
    :param test_benign_file_path:
    :param test_malicious_file_path:
    :param result_file_path:
    :param n_jobs:
    :return:
    """

    regex = _load_regex_list(input_file_path)
    benign_urls = _load_test_data(test_benign_file_path)
    malicious_urls = _load_test_data(test_malicious_file_path)

    benign_urls_plus = []
    for url in benign_urls:
        worker = UrlNormalize(url)
        benign_urls_plus.append(worker.get_domain_path_url())

    temp_res = Parallel(
        n_jobs=n_jobs)(
        delayed(_check_performance)(
            regex,
            benign_urls_plus,
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


def url_regex_publish(result_file_path,
                  publish_file_path,
                  publish_fp_thresh=PUBLISH_FP_THRESH,
                  publish_tp_thresh=PUBLISH_TP_THRESH,
                  publish_ratio=PUBLISH_RATIO):
    """
    :param result_file_path: regex fp tp result file path
    :param publish_file_path: final publish regex file path
    :param publish_fp_thresh:
    :param publish_tp_thresh:
    :param publish_ratio:
    :param publish_ratio_tp_thresh:
    :return:
    """
    df = _load_check_result(result_file_path)
    df["ratio"] = (df.fp + 0.01) / (df.tp + 0.01)
    # dump regular expression with 0 fp
    df = df.loc[df.ratio < publish_ratio]
    df = df.loc[df.fp <= publish_fp_thresh]
    df = df.loc[df.tp >= publish_tp_thresh]
    
    regex_list_publish = list(set(df.regex))
    logger.debug("regular expression publish\t%d" % len(regex_list_publish))
    # dump regular expressions
    _dump_regex_list(regex_list_publish, publish_file_path)
    return df


# evaluate the regex extracted from the list
def regex_evaluate(publish_file_path,
                   test_malicious_file_path):
    malicious_urls = _load_test_data(test_malicious_file_path)
    regex_list = _load_regex_list(publish_file_path)
    malicious_urls_hit = set()
    for index, regex in enumerate(regex_list):
        for url in malicious_urls:
            if _url_regex_match(regex, url):
                malicious_urls_hit.add(url)
    hit_percentage = float(len(malicious_urls_hit)) / \
        float(len(set(malicious_urls))) * 100
    logger.debug("hit percentage: %f" % hit_percentage)


# support function for malicious_url_predict
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
    for index, regex in enumerate(regex_list[start_index: end_index]):
        hit = []
        for url in test_urls:
            if _url_regex_match(regex, url):
                hit.append(url)
        print "batch index %d\tsample index %d\thit url %d\t%s" \
            %(batch_index, index, len(hit), regex)
        if len(hit) != 0:
            res[regex] = hit
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
    test_urls = _load_test_data(input_file_path)
    # preprocess
    test_urls_map = defaultdict(list)
    for url in test_urls:
        worker = UrlNormalize(url)
        test_urls_map[worker.get_domain_path_url()].append(url)

    # predict part
    predict_res_list = Parallel(
        n_jobs=n_jobs)(
        delayed(_core_predict)(
            regex,
            test_urls_map.keys(),
            index,
            n_jobs) for index in range(n_jobs))
    # precess the result
    predict_malicious = []
    predict_dict = dict()
    for predict_res in predict_res_list:
        for key in predict_res:
            for i in predict_res[key]:
                predict_malicious.extend(test_urls_map[i])
            predict_dict[key] = predict_res[key]

    predict_dict_final = dict()
    for k, v in predict_dict.iteritems():
        predict_dict_final[k] = [test_urls_map[_] for _ in v]
    return predict_malicious, predict_dict_final
