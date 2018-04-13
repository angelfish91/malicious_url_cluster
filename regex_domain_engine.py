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


from logger import logger
from config import cfg
from lib.max_substring import maxsubstring


DOMAIN_LEVEL_FREQUENCY_THRESH = cfg.DOMAIN_LEVEL_FREQUENCY_THRESH
RANDOM_LEVEL_SAMPLE_ROUND = cfg.RANDOM_LEVEL_SAMPLE_ROUND
RANDOM_LEVEL_SAMPLE_UPBOUND = cfg.RANDOM_LEVEL_SAMPLE_UPBOUND
RANDOM_LEVEL_SAMPLE_RATIO = cfg.RANDOM_LEVEL_SAMPLE_RATIO

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
                score_list.append(sum([_domain_regex_match(regex, _) for _ in sub_level_domain_list]))
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
    logger.debug("%s" %str(level_tree))
    logger.debug(regex)
    return regex


# extract domain regex
def domain_regex_extract(clusters):
    level_tree_list = [_build_domain_level_tree(_) for _ in clusters]
    regex_list = []
    for level_tree in level_tree_list:
        regex_list.append(_build_domain_regex(level_tree))
    return regex_list
    










