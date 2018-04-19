#!/usr/bin/evn python2.7
# -*- coding: utf-8 -*-
"""
域名正则表达式提取模块
"""


import re
import time
import random
from collections import Counter, defaultdict
from joblib import Parallel, delayed

from urlnormalize import UrlNormalize
from logger import logger
from config import cfg
from lib.max_substring import maxsubstring
from file_io import _load_cluster_data, _dump_regex_list, _load_regex_list, _load_test_data, \
    _dump_check_result, _load_check_result

N_JOBS = cfg.GLOBAL_N_JOBS

DOMAIN_CLUSTER_SIZE_THRESH = cfg.DOMAIN_CLUSTER_SIZE_THRESH
# sub level domain extract
DOMAIN_LEVEL_FREQUENCY_THRESH = cfg.DOMAIN_LEVEL_FREQUENCY_THRESH
TOP_DOMAIN_LEVEL_COUNT_THRESH = cfg.TOP_DOMAIN_LEVEL_COUNT_THRESH
SECOND_DOMAIN_LEVEL_COUNT_THRESH = cfg.SECOND_DOMAIN_LEVEL_COUNT_THRESH
# sampling
RANDOM_LEVEL_SAMPLE_ROUND = cfg.RANDOM_LEVEL_SAMPLE_ROUND
RANDOM_LEVEL_SAMPLE_UPBOUND = cfg.RANDOM_LEVEL_SAMPLE_UPBOUND
RANDOM_LEVEL_SAMPLE_RATIO = cfg.RANDOM_LEVEL_SAMPLE_RATIO

SMALL_CLUSTER_SIZE = cfg.SMALL_CLUSTER_SIZE
BIG_CLUSTER_SIZE = cfg.BIG_CLUSTER_SIZE

# domain regex publish
PUBLISH_DOMAIN_SCORE_THRESH = cfg.PUBLISH_DOMAIN_SCORE_THRESH
PUBLISH_DOMAIN_RATIO = cfg.PUBLISH_DOMAIN_RATIO
PUBLISH_DOMAIN_FP_THRESH = cfg.PUBLISH_DOMAIN_FP_THRESH
PUBLISH_DOMAIN_TP_THRESH = cfg.PUBLISH_DOMAIN_TP_THRESH


# analysis each level of domain, keep high frequent level
def _calc_regex_uncertainty(regex):
    uc1 = re.compile("\{\d{1,}\,(\d{1,})\}")
    uc1 = uc1.findall(regex)
    if len(uc1) != 0:
        uc1 = max([int(_) for _ in uc1])
    else:
        uc1 = 0
    uc2 = re.compile("\{(\d{1,})\}")
    uc2 = uc2.findall(regex)
    if len(uc2) != 0:
        uc2 = max([int(_) for _ in uc2])
    else:
        uc2 = 0
    uc = max(uc1, uc2)
    uc = len(regex) / float(uc)
    return uc


def _calc_pattern_len(pattern, regex):
    pattern = re.compile(pattern)
    return sum([len(_) for _ in pattern.findall(regex)])


def _calc_regex_score(regex):
    p1 = re.compile("\,(\d{1,})\}")
    p2 = re.compile("\{(\d{1,})\}")
    uncertain = p1.findall(regex)
    uncertain += p2.findall(regex)
    uncertain = sum([int(_) for _ in uncertain])
    certain = len(regex)
    dot = re.compile("\\.")
    dot = dot.findall(regex)
    certain -= len(dot) * 2
    pattern_list = ["\[A-Za-z\]\{\d{1,}\,\d{1,}\}",
                    "\[A-Za-z\]\{\d{1,}\}",
                    "\\\d\{\d{1,}\}",
                    "\\\d\{\d{1,}\,\d{1,}\}",
                    "\\\w\{\d{1,}\}",
                    "\\\w\{\d{1,}\,\d{1,}\}",
                    "\[\^\\\.\]\{\d{1,}\,\d{1,}\}",
                    "\[\^\\\.\]\{\d{1,}\}",
                    "\(\?:",
                    "\)",
                    "\|",
                    "com\\\.br",
                    "com\\\.cn",
                    "com\\\.uk",
                    "com\\\.au"]

    for i in pattern_list:
        certain -= _calc_pattern_len(i, regex)
    certain -= 2
    if certain < 0:
        certain = 0
    return float(certain) / (certain + uncertain)


# deduplicate regular expression
def regex_deduplicate(regex_list):
    pattern_1 = "\{(\d{1,})\,(\d{1,})\}"
    pattern_1 = re.compile(pattern_1)
    pattern_2 = "\{(\d{1,})\}"
    pattern_2 = re.compile(pattern_2)
    done = []
    regex_done = []
    for n, rex1 in enumerate(regex_list):
        if n in done:
            continue
        p1_res1 = pattern_1.search(rex1)
        p2_res1 = pattern_2.search(rex1)
        if p1_res1 is not None:
            rex1_fount = rex1[:p1_res1.span()[0]]
            rex1_back = rex1[p1_res1.span()[1]:]
            rex1_str = rex1_fount + rex1_back
            rex1_lo = int(p1_res1.groups()[0])
            rex1_hi = int(p1_res1.groups()[1])
        elif p2_res1 is not None:
            rex1_fount = rex1[:p2_res1.span()[0]]
            rex1_back = rex1[p2_res1.span()[1]:]
            rex1_str = rex1_fount + rex1_back
            rex1_lo = rex1_hi = int(p2_res1.groups()[0])
        else:
            regex_done.append(rex1)
            continue
        for m, rex2 in enumerate(regex_list[n + 1:]):
            p1_res2 = pattern_1.search(rex2)
            p2_res2 = pattern_2.search(rex2)
            if p1_res2 is not None:
                rex2_fount = rex2[:p1_res2.span()[0]]
                rex2_back = rex2[p1_res2.span()[1]:]
                rex2_str = rex2_fount + rex2_back
                if rex2_str == rex1_str:
                    done.append(n + m + 1)
                    rex1_lo = min(int(p1_res2.groups()[0]), rex1_lo)
                    rex1_hi = max(int(p1_res2.groups()[1]), rex1_hi)
            elif p2_res2 is not None:
                rex2_fount = rex2[:p2_res2.span()[0]]
                rex2_back = rex2[p2_res2.span()[1]:]
                rex2_str = rex2_fount + rex2_back
                if rex2_str == rex1_str:
                    done.append(n + m + 1)
                    rex1_lo = min(int(p2_res2.groups()[0]), rex1_lo)
                    rex1_hi = max(int(p2_res2.groups()[0]), rex1_hi)
        if rex1_lo != rex1_hi:
            regex_done.append("%s{%d,%d}%s" %
                              (rex1_fount, rex1_lo, rex1_hi, rex1_back))
        else:
            regex_done.append("%s{%d}%s" % (rex1_fount, rex1_lo, rex1_back))
    return regex_done


def _domain_sub_level_analysis(domains):
    size = len(domains)
    each_level_dict = defaultdict(list)
    for domain in domains:
        for level, each_level in enumerate(domain.split(".")[::-1]):
            each_level_dict[level].append(each_level)

    each_level_tree = defaultdict(list)
    for level, name_list in each_level_dict.iteritems():
        for name, name_count in Counter(name_list).iteritems():
            if float(name_count) / float(size) > DOMAIN_LEVEL_FREQUENCY_THRESH:
                each_level_tree[level].append(name)
            elif level == 0 and name_count > TOP_DOMAIN_LEVEL_COUNT_THRESH:
                each_level_tree[level].append(name)
            elif level == 1 and name_count > SECOND_DOMAIN_LEVEL_COUNT_THRESH:
                each_level_tree[level].append(name)

    return each_level_dict, each_level_tree


# search regex in sub level domain
def _sub_level_regex_extract(strings):
    max_substring_list = maxsubstring(strings, 2)
    pattern = ""
    for index in range(len(max_substring_list) + 1):
        sub_strings = []
        fut_strings = []
        if index != len(max_substring_list):
            max_substring = max_substring_list[index]
            for string in strings:
                sub_strings.append(
                    string.replace(
                        max_substring,
                        '\t').split('\t')[0])
                if index != len(max_substring_list):
                    fut_strings.append(
                        string.replace(
                            max_substring,
                            '\t').split('\t')[1])
        else:
            sub_strings = strings
        len_list = [len(_) for _ in sub_strings]
        size_max = max(len_list)
        size_min = min(len_list)
        if size_max == 0:
            pattern += ""
        elif sum([_.isdigit() or _ == "" for _ in sub_strings]) == len(strings):
            if size_max == size_min:
                pattern += "\d{%d}" % size_max
            else:
                pattern += "\d{%d,%d}" % (size_min, size_max)
        elif sum([_.isalpha() or _ == "" for _ in sub_strings]) == len(strings):
            if size_max == size_min:
                pattern += "[A-Za-z]{%d}" % size_max
            else:
                pattern += "[A-Za-z]{%d,%d}" % (size_min, size_max)
        elif sum([_.isalnum() or _ == "" for _ in sub_strings]) == len(strings):
            if size_max == size_min:
                pattern += "\w{%d}" % size_max
            else:
                pattern += "\w{%d,%d}" % (size_min, size_max)
        else:
            if size_max == size_min:
                pattern += "[^\.]{%d}" % size_max
            else:
                pattern += "[^\.]{%d,%d}" % (size_min, size_max)
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


def _filter_domain_by_sub_level_domain_count(cluster):
    fcluster = [_ for _ in cluster if _.count(
        ".") == cluster[0].count(".")]
    return fcluster


# build domain level tree
def _build_domain_level_tree(cluster):
    cluster = _filter_domain_by_sub_level_domain_count(cluster)
    level_dict, level_tree = _domain_sub_level_analysis(cluster)
    for level in level_dict:
        if level not in level_tree.keys() and (level == 0 or level == 1):
            return None
        if level not in level_tree.keys():
            sub_level_domain_list = level_dict[level]
            score_list = []
            regex_list = []
            for sample_round in range(RANDOM_LEVEL_SAMPLE_ROUND):
                sample_num = int(
                    len(sub_level_domain_list) *
                    RANDOM_LEVEL_SAMPLE_RATIO)
                if sample_num > RANDOM_LEVEL_SAMPLE_UPBOUND:
                    sample_num = RANDOM_LEVEL_SAMPLE_UPBOUND
                if sample_num == 0:
                    sample_num = len(sub_level_domain_list)
                sample = random.sample(sub_level_domain_list, sample_num)
                try:
                    regex = _sub_level_regex_extract(sample)
                except Exception as e:
                    regex = "[^\.]{%d,%d}" % (
                        min([len(_) for _ in sample]), max([len(_) for _ in sample]))
                    logger.warning("regex extract fail! %s" % str(e))
                regex_list.append(regex)
                score_list.append(
                    sum([_sub_level_domain_regex_match(regex, _) for _ in sub_level_domain_list]))
            max_score_index = score_list.index(max(score_list))
            regex = regex_list[max_score_index]
            level_tree[level].append(regex)
    return level_tree


# get domain regular expression
def _build_domain_regex(level_tree):
    regex_list = []
    for i in range(len(level_tree) - 1, -1, -1):
        if len(level_tree[i]) == 1:
            regex_list.append(level_tree[i][0])
        else:
            regex = "|".join(level_tree[i])
            regex_list.append("(:?" + regex + ")")
    regex = "\.".join(regex_list)
    regex = "^" + regex + "$"
    logger.debug("%s" % str(level_tree))
    logger.debug(regex)
    return regex


# extract domain regex
def domain_regex_extract(input_file_path,
                         output_file_path,
                         cluster_size_thresh=DOMAIN_CLUSTER_SIZE_THRESH,
                         dump=True):
    """
    :param input_file_path:
    :param output_file_path:
    :param cluster_size_thresh:
    :param dump:
    :return:
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

    # filter small clusters
    cluster_list = [_ for _ in cluster_list if len(
        _) >= cluster_size_thresh]
    # build sub domain level tree
    level_tree_list = [_build_domain_level_tree(_) for _ in cluster_list]
    level_tree_list = [_ for _ in level_tree_list if _ is not None]
    regex_list = []
    for level_tree in level_tree_list:
        regex_list.append(_build_domain_regex(level_tree))

    regex_list = list(set(regex_list))
    logger.debug("extract regex count:\t%d" % len(regex_list))
    regex_list = regex_deduplicate(regex_list)
    logger.debug(
        "extract regex count:\t%d\t(after deduplicate)" %
        len(regex_list))
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
        benign_res = sum([_domain_regex_match(regex, _) for _ in benign_urls])
        malicious_res = sum([_domain_regex_match(regex, _)
                             for _ in malicious_urls])
        benign_res_set = sum([_domain_regex_match(regex, _)
                              for _ in set(benign_urls)])
        malicious_res_set = sum([_domain_regex_match(regex, _)
                                 for _ in set(malicious_urls)])
        print "batch index %d\tsample index %d\tFP(url) %d\tFP(domain) %d\tTP(url) %d\tTP(domain) %d\t%s" \
              % (batch_index, index, benign_res, benign_res_set, malicious_res, malicious_res_set, regex)
        res.append((benign_res, malicious_res))
    return res


# regular expression check and publish
def domain_regex_check(input_file_path,
                       test_benign_file_path,
                       test_malicious_file_path,
                       result_file_path,
                       n_jobs=N_JOBS):
    """
    对抽取出的正则表达式进行黑白数据的性能评估
    测试黑白数据为原始的URL
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
    # 测试用数据处理
    malicious_urls_plus = []
    for url in malicious_urls:
        worker = UrlNormalize(url)
        malicious_urls_plus.append(worker.get_hostname())

    benign_urls_plus = []
    for url in benign_urls:
        worker = UrlNormalize(url)
        benign_urls_plus.append(worker.get_hostname())

    temp_res = Parallel(
        n_jobs=n_jobs)(
        delayed(_check_performance)(
            regex,
            benign_urls_plus,
            malicious_urls_plus,
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


def domain_regex_publish(result_file_path,
                         publish_file_path,
                         publish_score_thresh=PUBLISH_DOMAIN_SCORE_THRESH,
                         publish_ratio=PUBLISH_DOMAIN_RATIO,
                         publish_fp_thresh=PUBLISH_DOMAIN_FP_THRESH,
                         publish_tp_thresh=PUBLISH_DOMAIN_TP_THRESH):
    df = _load_check_result(result_file_path)
    df = df.drop_duplicates("regex")
    df["score"] = [_calc_regex_score(_) for _ in df.regex]
    df = df.loc[df.score > publish_score_thresh]
    df["ratio"] = (df.fp + 0.01) / (df.tp + 0.01)
    df = df.loc[df.ratio < publish_ratio]
    df = df.loc[df.fp <= publish_fp_thresh]
    df = df.loc[df.tp >= publish_tp_thresh]

    regex_list_publish = list(df.regex)
    regex_list_publish = regex_deduplicate(regex_list_publish)
    logger.debug("regular expression publish\t%d" % len(regex_list_publish))
    # dump regular expressions
    _dump_regex_list(regex_list_publish, publish_file_path)
    return df


# support function for malicious_domain_predict
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
            if _domain_regex_match(regex, url):
                hit.append(url)
        print "batch index %d\tsample index %d\thit domain %d\t%s" \
            % (batch_index, index, len(hit), regex)
        if len(hit) != 0:
            res[regex] = hit
    return res


# predict for unknown domain
def malicious_domain_predict(input_file_path,
                             regex_file_path,
                             n_jobs=N_JOBS):
    """
    :param input_file_path:
    :param regex_file_path:
    :param n_jobs:
    :return: predict_malicious [list], predict_dict [dict], predict_dict_detail [dict]
    """
    regex = _load_regex_list(regex_file_path)
    test_urls = _load_test_data(input_file_path)
    # 预处理，构建了域名与原始URL的字典
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
        for regex in predict_res:
            predict_malicious.extend([test_urls_map[_]
                                      for _ in predict_res[regex]])
            predict_dict[regex] = predict_res[regex]

    predict_malicious = [__ for _ in predict_malicious for __ in _]
    predict_dict_detail = dict()
    for regex, hit_domains in predict_dict.iteritems():
        predict_dict_detail[regex] = [test_urls_map[_] for _ in hit_domains]
    return predict_malicious, predict_dict, predict_dict_detail
