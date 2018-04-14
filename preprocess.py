#!/usr/bin/evn python2.7
# -*- coding: utf-8 -*-

"""
URL数据统计与预处理模块
数据统计
1. 纯为域名的url
2. 拥有path的url
3. 带参数的url
"""
import os
import sys
import pandas as pd
from collections import defaultdict
from joblib import Parallel, delayed
import socket
import urllib

from config import cfg
import urlnormalize
from logger import logger

N_JOBS = cfg.GLOBAL_N_JOBS

def _core_get_ip(url):
    worker = urlnormalize.UrlNormalize(url)
    domain = worker.get_hostname()
    url_plus = worker.get_quote_plus_url(url)       
    try:
        ip = socket.gethostbyname(domain)
    except:
        ip = "unknown"
    return ip, url_plus


def url_map_ip_analysis(urls, n_jobs = N_JOBS):
    res = Parallel(n_jobs=n_jobs)(delayed(_core_get_ip)(url) for url in urls)
    ip_url_map = defaultdict(list)
    for ip, url in res:
        ip_url_map[ip].append(url)
    return ip_url_map
        

def url_classify_analysis(urls):
    only_domain = set()
    url_with_path = set()
    url_with_param = set()
    for url in urls:
        if len(url) <= cfg.SINGLE_REGEX_SIZE:
            continue
        worker = urlnormalize.UrlNormalize(url)
        if worker.url_is_only_domain():
            only_domain.add(woker.get_hostname())
        elif len(worker.get_params()) == 0:
            url_with_path.add(worker.get_quote_plus_url(url))
        else:
            url_with_param.add(worker.get_quote_plus_url(url))
    return list(only_domain), list(url_with_path), list(url_with_param)


def url_split_analysis(urls):
    domain = set()
    path = set()
    for url in urls:
        worker = urlnormalize.UrlNormalize(url)
        domain.add(worker.get_hostname())
        path.add(urllib.quote_plus(worker.get_path()))
    return domain, path
    

def url_path_analysis(urls):
    res = defaultdict(list)
    for url in urls:
        worker = urlnormalize.UrlNormalize(url)
        res[len(worker.get_dir_list())].append(url)
    res_len = dict()
    for k, v in res.iteritems():
        res_len[k] = len(v)
    return res, res_len



