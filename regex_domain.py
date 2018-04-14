#!/usr/bin/evn python2.7
# -*- coding: utf-8 -*-
"""
域名正则表达式提取模块
"""

import os
import sys
import re
import json
import time
import random
import pandas as pd
from collections import Counter, defaultdict
from joblib import Parallel, delayed

from urlnormalize import UrlNormalize
from logger import logger
from config import cfg
from lib.max_substring import maxsubstring
from regex_url import _load_cluster_data, _dump_regex_list, _load_regex_list, _load_test_data


N_JOBS = cfg.GLOBAL_N_JOBS

DOMAIN_CLUSTER_SIZE_THRESH = cfg.DOMAIN_CLUSTER_SIZE_THRESH
DOMAIN_LEVEL_FREQUENCY_THRESH = cfg.DOMAIN_LEVEL_FREQUENCY_THRESH
RANDOM_LEVEL_SAMPLE_ROUND = cfg.RANDOM_LEVEL_SAMPLE_ROUND
RANDOM_LEVEL_SAMPLE_UPBOUND = cfg.RANDOM_LEVEL_SAMPLE_UPBOUND
RANDOM_LEVEL_SAMPLE_RATIO = cfg.RANDOM_LEVEL_SAMPLE_RATIO

SMALL_CLUSTER_SIZE = cfg.SMALL_CLUSTER_SIZE
BIG_CLUSTER_SIZE = cfg.BIG_CLUSTER_SIZE

# analysis each level of domain, keep high frequent level
def _domain_sub_level_analysis(domains, thresh = 0.1):
    size = len(domains)
    each_level_dict = defaultdict(list)
    for domain in domains:
        for level, each_level in enumerate(domain.split(".")[::-1]):
            each_level_dict[level].append(each_level)
    
    each_level_tree = defaultdict(list)
    for level, name_list in each_level_dict.iteritems():
        for name, name_count in Counter(name_list).iteritems():
            if float(name_count)/float(size) > DOMAIN_LEVEL_FREQUENCY_THRESH:
                each_level_tree[level].append(name)
            
    return each_level_dict, each_level_tree


# search regex in sub level domain
def _sub_level_regex_extract(strings):            
    max_substring_list = maxsubstring(strings, 2)
    pattern = ""
        
    for index in range(len(max_substring_list) + 1):
        if index != len(max_substring_list):
            max_substring = max_substring_list[index]
        sub_strings = []
        fut_strings = []
        if 'max_substring' in locals():
            for string in strings:
                sub_strings.append(string.replace(max_substring, '\t').split('\t')[0])
                if index != len(max_substring_list):
                    fut_strings.append(string.replace(max_substring, '\t').split('\t')[1])
        else:
            sub_strings = strings
        len_list = [len(_) for _ in sub_strings]
        size_max = max(len_list)
        size_min = min(len_list)
        if size_max == 0:
            pattern += ""
        elif sum([_.isdigit() or _ == "" for _ in sub_strings]) == len(strings):
            if size_max == size_min:
                pattern += "\d{%d}" %size_max
            else:
                pattern += "\d{%d,%d}" %(size_min, size_max)
        elif sum([_.isalpha() or _ == "" for _ in sub_strings]) == len(strings):
            if size_max == size_min:
                pattern += "[A-Za-z]{%d}" %size_max
            else:
                pattern += "[A-Za-z]{%d,%d}" %(size_min, size_max)
        elif sum([_.isalnum() or _ == "" for _ in sub_strings]) == len(strings):
            if size_max == size_min:
                pattern += "\w{%d}" %size_max
            else:
                pattern += "\w{%d,%d}" %(size_min, size_max)
        else:
            if size_max == size_min:
                pattern += "(.){%d}" %size_max
            else:
                pattern += "(.){%d,%d}" %(size_min, size_max)
        if index != len(max_substring_list):
            pattern += max_substring
        strings = fut_strings
    return pattern


# sub level domain match
def _sub_level_domain_regex_match(regex, domain):
    regex = "^" + regex + "$"
    pattern = re.compile(regex)
    if pattern.match(domain):
        return True
    return False


# domain match
def _domain_regex_match(regex, domain):
    pattern = re.compile(regex)
    if pattern.match(domain):
        return True
    return False


# build domain level tree
def _build_domain_level_tree(cluster):
    level_dict, level_tree = _domain_sub_level_analysis(cluster)
    for level in level_dict:
        if level not in level_tree.keys():
            sub_level_domain_list = level_dict[level]
            score_list = []
            regex_list = []
            for sample_round in range(RANDOM_LEVEL_SAMPLE_ROUND):
                sample_num = int(len(cluster) * RANDOM_LEVEL_SAMPLE_RATIO)
                if sample_num > RANDOM_LEVEL_SAMPLE_UPBOUND:
                    sample_num == RANDOM_LEVEL_SAMPLE_UPBOUND
                sample = random.sample(sub_level_domain_list, sample_num)
                regex = _sub_level_regex_extract(sample)
                regex_list.append(regex)
                score_list.append(sum([_sub_level_domain_regex_match(regex, _) for _ in sub_level_domain_list]))
            max_score_index = score_list.index(max(score_list))
            regex = regex_list[max_score_index]
            level_tree[level].append(regex)
    return level_tree


# get doamin regular expression
def _build_domain_regex(level_tree):
    regex_list = []
    for i in range(len(level_tree)-1, -1, -1):
        if len(level_tree[i]) == 1:
            regex_list.append(level_tree[i][0])
        else:
            regex = "|".join(level_tree[i])
            regex_list.append("(:?"+regex+")")
    regex = "\.".join(regex_list)
    regex = "^" + regex + "$"
    logger.debug("%s" %str(level_tree))
    logger.debug(regex)
    return regex


# extract domain regex
def domain_regex_extract(input_file_path,
                  output_file_path, dump = True):
    start_time = time.time()
    clusters = _load_cluster_data(input_file_path)
    cluster_size = [len(_) for _ in clusters]
    # log detail info of cluster on console
    logger.debug("total cluster num:\t%d" % len(clusters))
    logger.debug("big cluster:\t%d" %
                 len([1 for i in cluster_size if i >= BIG_CLUSTER_SIZE]))
    logger.debug("small cluster:\t%d" % len(
        [1 for i in cluster_size if SMALL_CLUSTER_SIZE <= i < BIG_CLUSTER_SIZE]))
    logger.debug("single one:\t%d" % len([1 for i in cluster_size if i == 1]))
    logger.debug("cluster size detail:\t%s" % str(Counter(cluster_size)))
    
    
    # filter small clusters
    
    clusters = [_ for _ in clusters if len(_) >= DOMAIN_CLUSTER_SIZE_THRESH]
    # build sub domain level tree
    level_tree_list = [_build_domain_level_tree(_) for _ in clusters]
    regex_list = []
    for level_tree in level_tree_list:
        regex_list.append(_build_domain_regex(level_tree))
        
    logger.debug("extract regex count:\t%d" % len(regex_list))
    logger.debug("extract regex time cost:\t%f" % (time.time() - start_time))
    if dump:
        _dump_regex_list(regex_list, output_file_path)
    else:
        return regex_list
    

    
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
            if _domain_regex_match(i, url):
                hit.append(url)
        print(
            "batch index",
            batch_index,
            "sample index",
            index,
            "hit", len(hit)
        )
        if len(hit) != 0:
            res[i] = hit
    return res


# predict for unknown domain
def malicious_domain_predict(input_file_path,
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
        test_urls_map[worker.get_hostname()].append(url)    
    
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
        for k in predict_res:
            predict_malicious.extend([test_urls_map[_] for _ in predict_res[k]])
            predict_dict[k] = predict_res[k]
    
    predict_malicious = [__ for _ in predict_malicious for __ in _]
    predict_dict_detail = dict()
    for k, v in predict_dict.iteritems():
        predict_dict_detail[k] = [test_urls_map[_] for _ in v]
    return predict_malicious, predict_dict, predict_dict_detail










