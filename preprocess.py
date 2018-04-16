#!/usr/bin/evn python2.7
# -*- coding: utf-8 -*-

"""
URL数据统计与预处理模块
"""
import re
import socket
import urllib
from collections import defaultdict
from joblib import Parallel, delayed


from config import cfg
import urlnormalize
from logger import logger

N_JOBS = cfg.GLOBAL_N_JOBS


# support function for make_ip_url_map
def _core_get_ip(url):
    """
    :param url:
    :return:
    """
    worker = urlnormalize.UrlNormalize(url)
    domain = worker.get_hostname()
    url_plus = worker.get_quote_plus_url()
    try:
        ip = socket.gethostbyname(domain)
    except Exception as e:
        logger.warning("%s ip request failed %s" % (domain, str(e)))
        ip = "unknown"
    return ip, url_plus


# get ip url map
def make_ip_url_map(urls, n_jobs=N_JOBS):
    """
    :param urls:
    :param n_jobs:
    :return: [dict]
    """
    res = Parallel(n_jobs=n_jobs)(delayed(_core_get_ip)(url) for url in urls)
    ip_url_map = defaultdict(list)
    for ip, url in res:
        ip_url_map[ip].append(url)
    return ip_url_map


# split url into only domain, url with path, url with param
def make_url_classification(urls):
    """
    :param urls:
    :return: only_domain [list], url_with_path [list], url_with_param [list]
    """
    only_domain, url_with_path, url_with_param = [], [], []
    for url in urls:
        if len(url) <= cfg.SINGLE_REGEX_SIZE:
            logger.debug("url length do not meet size req %d %s" %(cfg.SINGLE_REGEX_SIZE, url))
            continue
        worker = urlnormalize.UrlNormalize(url)
        if worker.url_is_only_domain():
            only_domain.append(worker.get_hostname())
        elif len(worker.get_params()) == 0:
            url_with_path.append(worker.get_domain_path_url())
        else:
            url_with_param.append(worker.get_quote_plus_url())
    return only_domain, url_with_path, url_with_param


# split url into domain and path
def make_url_split(urls):
    """
    :param urls:
    :return: domain [list], path [list]
    """
    domain, path = set(), set()
    for url in urls:
        worker = urlnormalize.UrlNormalize(url)
        domain.add(worker.get_hostname())
        path.add(urllib.quote_plus(worker.get_path()))
    return list(domain), list(path)


# count url path level
def make_url_path_level_count(urls):
    """
    :param urls:
    :return: res [defaultdict(list)], counter [dict]
    """
    res = defaultdict(list)
    for url in urls:
        worker = urlnormalize.UrlNormalize(url)
        res[len(worker.get_dir_list())].append(url)
    counter = dict()
    for k, v in res.iteritems():
        counter[k] = len(v)
    return res, counter




def check_domain(domain):
    """
    检测一个字符串是否是域名
    :param domain: 待检测的字符串
    :return: bool，如果该字符串是域名返回True，否则返回False
    """
    regex_domain = re.compile('(?i)^([a-z0-9]+(-[a-z0-9]+)*\.)+[a-z]{2,}$')
    if regex_domain.match(domain) is None:
        return False
    return True


def check_ip(check_str):
    """
    检测一个字符串是否是IP
    :param check_str: 待检测的字符串
    :return: bool，如果该字符串是IP返回True，否则返回False
    """
    regex_ip = re.compile(
        '^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$')
    if regex_ip.match(check_str) is None:
        return False
    return True


def check_url_ml(url):
    """
    检测一个字符串是否是url
    :param url:
    :return: bool，如果该字符串是url返回True，否则返回False
    """
    normalized_url = UrlNormalize(url)
    domain = normalized_url.get_hostname()
    if not check_domain(domain) and not check_ip(domain):
        return False

    url_path = normalized_url.get_path()
    if url_path != '' and url_path.find('/') == -1:
        return False

    return True

