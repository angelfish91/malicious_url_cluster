#!/usr/bin/evn python2.7
# -*- coding: utf-8 -*-
"""
build url forest
"""
import time
import pandas as pd
import tldextract

from urlnormalize import UrlNormalize
from preprocess import check_domain


def build_domain_forest(domain_list, batch_size = 50000):
    """
    build domain tree for detection
    :param domain_list: list of domain
    :domain_tree:
    :return: domain tree [dict]
    """
    domain_forest = []
    batch_num = len(domain_list)/batch_size
    if len(domain_list)%batch_size !=0:
        batch_num += 1
    for batch_index in range(batch_num):
        print "building tree index:%d" %batch_index
        domain_tree = dict()
        for domain in domain_list[batch_size*batch_index:batch_size*(batch_index+1)]:
            domain_tokens = domain.split(".")[::-1]
            sub_tree = domain_tree
            for token in domain_tokens:
                if token not in sub_tree.keys():
                    tmp_tree = {token:{}}
                    sub_tree.update(tmp_tree)
                    sub_tree = tmp_tree[token]
                else:
                    sub_tree = sub_tree[token]
        domain_forest.append(domain_tree)
    return domain_forest


def _path_count(tree_node):
    if len(tree_node) == 0:
        return 1
    return sum([ _path_count(tree_node[_]) for _ in tree_node])

def _analyze_domain_tree(tree_node, domain, depth = None, verbose = False):
    domain_tokens = domain.split(".")[::-1]
    if depth is None:
        depth = len(domain_tokens)
    level, match = 0, 0
    for level, token in enumerate(domain_tokens):
        level +=1
        if level < depth:
            if token in tree_node.keys():
                tree_node = tree_node[token]
            else:
                break
        if level == depth:
            if token in tree_node.keys():
                tree_node = tree_node[token]
                match = _path_count(tree_node)    
            break
    return (level, match)

def _analyze_domain_forest(domain_forest, domain, depth = None, verbose = False):
    tree_res_list = [ _analyze_domain_tree(_, domain, depth) for _ in domain_forest]
    res = sum([_[1] for _ in tree_res_list if _[0] == depth])
    if verbose:
        print tree_res_list
    return res

def analyze_domain(safe_forest, mal_forest, domain, check=False, verbose = False):
    # check whether the domain in white list or black list
    domain_tokens = domain.split(".")[::-1]
    depth = len(domain_tokens)
    if check:
        if _analyze_domain_forest(safe_forest, domain, depth):
            return 0
        elif _analyze_domain_forest(mal_forest, domain, depth):
            return 1
        else:
            return -1
    # count the up level domain num
    # limit = 2
    # if domain_tokens[0] in ["cn", "fr", "au", "uk", "us", "br", "pl", "tr",\
    #                  "ua", "ph", "ar", "in", "mx", "rs", "ke", "ru" \
    #                  "es", "vn", "pk", "id", "at", "pe", "ng"]:
    #    limit = 3
    extract_domain = tldextract.extract(domain).domain
    if extract_domain in domain_tokens:
        limit = domain_tokens.index(extract_domain) + 1
    else:
        print domain
        return -1
    if depth>limit:
        safe_count = _analyze_domain_forest(safe_forest, domain, depth-1)
        mal_count = _analyze_domain_forest(mal_forest, domain, depth-1)
        if verbose:
            print safe_count, mal_count
        if safe_count == 0 and mal_count == 0:
            return -1
        else:
            return mal_count/float(safe_count+0.01)
        
    else:
        return -1
    

def analyze_domain_batch(url_list, safe_forest, mal_forest, verbose = False):
    domain_list = []
    for url in url_list:
        worker = UrlNormalize(url)
        domain_list.append(worker.get_hostname())
    res_list = []
    cache_dict = {}
    for n, domain in enumerate(domain_list):
        domain_split = domain.split(".", 1)
        if len(domain_split) > 1  and domain_split[1] in cache_dict:
            res = cache_dict[domain_split[1]]
        else:
            res = analyze_domain(safe_forest, mal_forest, domain)
            if len(domain_split)>1:
                cache_dict[domain_split[1]] = res 
        res_list.append(res)
        if verbose:
            if n%5000 == 0:
                print "step:%d\tuncertain:%d\tpredict_mal:%d" %(n, 
                sum([1 for _ in res_list if _ == -1]), sum([1 for _ in res_list if _ > 1]))
    df = pd.DataFrame({"url": url_list, "domain_reputation": res_list})
    return df

