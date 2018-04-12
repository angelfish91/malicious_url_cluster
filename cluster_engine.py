#!/usr/bin/evn python2.7
# -*- coding: utf-8 -*-

"""
聚类模块，包含如下功能
1. K-means 聚类
2. 基于字符串相似度聚类
"""
import os
import sys
import time
import json
import pandas as pd
import Levenshtein as ls
from collections import Counter
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

from logger import logger
from config import cfg
from vectorize import _load_vector_data, make_vectorize

N_CLUSTER_RATION = cfg.N_CLUSTER_RATION
N_JOBS = cfg.GLOBAL_N_JOBS
EDIT_DISTANCE_THRESH_LONG = cfg.EDIT_DISTANCE_THRESH_LONG
EDIT_DISTANCE_THRESH_SHORT = cfg.EDIT_DISTANCE_THRESH_SHORT
LONG_URL_THRESH = cfg.LONG_URL_THRESH
SHORT_URL_THRESH = cfg.SHORT_URL_THRESH


def _init_dump_cluster_data(file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)
        logger.debug("OLD DATA FIND! REMOVING\t%s" % file_path)


def _dump_cluster_data(file_path, single_cluster, index):
    with open(file_path, "a+") as fd:
        fd.write(json.dumps({index: single_cluster}) + '\n')


def _load_kmeans_hier_cluster_data(file_path):
    res = []
    try:
        with open(file_path, 'r') as fd:
            res_dict = json.loads(fd.read().strip())
        for i in range(len(res_dict)):
            res.append(res_dict[str(i)])
        logger.debug("K-means data has been loaded\t%s" % file_path)
    except Exception as e:
        logger.warning("%s\t FILE OPEN ERROR!\t%s" % (file_path, str(e)))
        sys.exit(0)
    return res


def _dump_kmeans_hier_cluster_data(data, file_path):
    if os.path.isfile(file_path):
        os.remove(file_path)
        logger.debug("OLD DATA FIND! REMOVING\t%s" % file_path)
    try:
        with open(file_path, 'w+') as fd:
            fd.write(json.dumps(data))
        logger.debug("cluster has been dump\t%s" % file_path)
    except Exception as e:
        logger.warning("%s\tFILE DUMP ERROR! %s" % (file_path, e))
        sys.exit(0)


# calculate two url distance
def _core_distance_check(base_url, comp_url, metric):
    """
    :param base_url: base url string to compare
    :param comp_url: second url string to compare with base url string
    :param metric: method use to calculate distance
    :return: true of false whether the comp_url is similar with base url
    """
    if metric == "distance":
        base_thresh = 6
        base_url_len = len(base_url)
        if base_url_len < SHORT_URL_THRESH:
            base_thresh = 6
        if SHORT_URL_THRESH <= base_url_len < LONG_URL_THRESH:
            base_thresh = int(base_url_len * EDIT_DISTANCE_THRESH_SHORT)
        if LONG_URL_THRESH <= base_url_len:
            base_thresh = int(base_url_len * EDIT_DISTANCE_THRESH_LONG)
        if ls.distance(base_url, comp_url) <= base_thresh:
            return comp_url
        return False
    if metric == "jaro_winkler":
        if ls.jaro_winkler(base_url, comp_url) == 1:
            return comp_url
    if metric == "jaro":
        if ls.jaro(base_url, comp_url) > 0.9:
            return comp_url
        return False


# make fine grained cluster using string distance algorithms
def make_string_distance_cluster(
        data,
        n_jobs=N_JOBS,
        metric="distance",
        file_path=cfg.CLUSTER_DISTANCE_DATA_PATH):
    """
    :param data: url list[list] or k-means results file path
    :param n_jobs: multi-process
    :param metric: algorithms used to metric the string distance
    :param file_path: output file
    :return: None
    """
    start_time = time.time()
    # normalize input
    assert isinstance(data, list) or isinstance(data, str)
    if isinstance(data, list) and isinstance(data[0], str):
        urls_list = [data]
    elif isinstance(data, list) and isinstance(data[0], list):
        urls_list = data
    else:
        urls_list = _load_kmeans_hier_cluster_data(data)
    # prepare the output file
    _init_dump_cluster_data(file_path)
    counter = 0
    # url distance cluster
    for batch_index, batch_urls in enumerate(urls_list):
        cluster = []
        in_cluster = set()
        # for each sub cluster of k-mean results
        for n, i in enumerate(batch_urls):
            if n in in_cluster:
                continue
            logger.debug("----------------%d------------------" % n)
            single_cluster = [i]
            res = Parallel(n_jobs=n_jobs)(delayed(_core_distance_check)(
                i, j, metric) for j in batch_urls[n + 1:] if j not in in_cluster)
            for m, j in enumerate(res):
                if j:
                    single_cluster.append(j)
                    in_cluster.add(n + 1 + m)
            in_cluster.add(i)
            # log result on console
            for line in single_cluster:
                logger.debug(str(line))
            logger.debug(
                ">>>>>>> BATCH:%d/%d\tTOTAL:%d\tDONE:%d" %
                (len(urls_list), batch_index + 1, len(batch_urls), len(in_cluster)))
            cluster.append(single_cluster)
            # dump result data
            counter += 1
            _dump_cluster_data(file_path, single_cluster, counter)
    logger.debug("string distance time cost:\t%f" % (time.time() - start_time))


def make_hier_cluster(
        data=cfg.VECTOR_DOMAIN_DATA,
        output_path=cfg.KMEANS_CLUSTER_DATA,
        cluster_num=None,
        dump=True):
    # normalize input
    assert isinstance(data, str) or isinstance(data, pd.DataFrame)
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = _load_vector_data(data)

    # calculate n cluster for K-means clustering
    total_data_size = len(df)
    n_clusters = int(total_data_size / N_CLUSTER_RATION)
    if cluster_num is not None:
        n_clusters = cluster_num
    # log n_clusters to console
    logger.debug(
        "beggin to make hierarchical cluster, total_data_size: %d n_clusters: %d" %
        (total_data_size, n_clusters))
    # exception
    if n_clusters == 0:
        return {0: list(df.index)}
    # clustering
    clustering = AgglomerativeClustering(linkage='complete',
                                         affinity="l1",
                                         n_clusters=n_clusters,
                                         compute_full_tree=False)
    clustering.fit(df.values)
    df["labels"] = clustering.labels_

    # dump results dict format
    res = dict()
    for i in range(n_clusters):
        df_single = df.loc[df["labels"] == i]
        res[i] = list(df_single.index)
    logger.debug("hcluster done!")
    if dump:
        _dump_kmeans_hier_cluster_data(res, file_path=output_path)
    else:
        return res


def make_kmeans_cluster(
        data=cfg.VECTOR_DOMAIN_DATA,
        output_path=cfg.KMEANS_CLUSTER_DATA,
        cluster_num=None,
        dump=True):
    # normalize input
    assert isinstance(data, str) or isinstance(data, pd.DataFrame)
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = _load_vector_data(data)
    # calculate n cluster for K-means clustering
    total_data_size = len(df)
    n_clusters = int(total_data_size / N_CLUSTER_RATION)
    if n_clusters < 3:
        n_clusters = 10

    # log n_clusters to console
    logger.debug(
        "beggin to make k-means cluster, total_data_size: %d n_clusters: %d" %
        (total_data_size, n_clusters))
    if cluster_num is not None:
        optimal_n_cluster = cluster_num
    else:
        # make k-means cluster exp  for optimal n_clusters
        calib = []
        for i in range(2, n_clusters):
            kmeans = KMeans(
                n_clusters=i,
                verbose=0,
                random_state=0,
                n_jobs=N_JOBS)
            kmeans.fit(df.values)
            score = metrics.calinski_harabaz_score(df.values, kmeans.labels_)
            calib.append(score)
            logger.debug(
                "Calinski-Harabasz Score for\t%d Cluster K-means\t%s" %
                (i, str(score)))

        optimal_n_cluster = calib.index(max(calib)) + 2
    logger.debug("optimal n_cluster\t%d" % optimal_n_cluster)

    # make k-means cluster with optimized n_clusters
    kmeans = KMeans(
        n_clusters=optimal_n_cluster,
        verbose=0,
        random_state=0,
        n_jobs=N_JOBS)
    kmeans.fit(df.values)
    df["labels"] = kmeans.labels_

    # dump results
    res = dict()
    for i in range(optimal_n_cluster):
        df_single = df.loc[df["labels"] == i]
        res[i] = list(df_single.index)
    logger.debug("k-means cluster done!")
    if dump:
        _dump_kmeans_hier_cluster_data(res, file_path=output_path)
    else:
        return res


# for mass urls training
def make_kmeans_cluster_mass(
        urls,
        domain=True,
        path=True,
        param=True,
        thresh=50000):
    assert isinstance(urls, list)
    df_vector = make_vectorize(urls, domain, path, param, dump=False)
    kmeans_res_dict = make_kmeans_cluster(
        df_vector, cluster_num=int(
            len(df_vector) / thresh), dump = False)
    logger.debug("Preliminary K-means clustering complete")
    kmeans_res_list = [kmeans_res_dict[_] for _ in kmeans_res_dict]
    # perform sub clustering for big cluster

    def kmeans_round(pre_cluster_list):
        sub_cluster_list = []
        flag = 0
        for index, pre_cluster in enumerate(pre_cluster_list):
            if len(pre_cluster) > thresh:
                flag += 1
                urls_tmp = pre_cluster_list.pop(index)
                df_vector_tmp = make_vectorize(
                    urls_tmp, domain, path, param, dump=False)
                kres = make_kmeans_cluster(
                    df_vector_tmp, cluster_num=3, dump=False)
                for k, v in kres.iteritems():
                    sub_cluster_list.append(v)
        pre_cluster_list += sub_cluster_list
        return pre_cluster_list, flag
    kmeans_res_list, flag = kmeans_round(kmeans_res_list)
    while flag != 0:
        kmeans_res_list, flag = kmeans_round(kmeans_res_list)
    return kmeans_res_list


def make_hier_cluster_mass(
        data,
        domain=True,
        path=True,
        param=True,
        output_path=None,
        dump=True):
    hcluster_res_list = []
    for urls in data:
        df_vector = make_vectorize(urls, domain, path, param, dump=False)
        hres = make_hier_cluster(df_vector, dump=False)
        for k, v in hres.iteritems():
            hcluster_res_list.append(v)
    # flter hier cluster result
    _cluster_filter(hcluster_res_list)
    if dump:
        hier_cluster_res_dict = dict()
        for index, v in enumerate(hcluster_res_list):
            hier_cluster_res_dict[index] = v
        _dump_kmeans_hier_cluster_data(hier_cluster_res_dict, output_path)
    else:
        return hcluster_res_list


def _cluster_filter(cluster_res_list, filter_limit=1):
    cluster_res_list = [_ for _ in cluster_res_list if type(_) == list]
    cluster_lens = [len(_) for _ in cluster_res_list]
    logger.debug("before filter %s" % str(Counter(cluster_lens)))
    cluster_res_list = [_ for _ in cluster_res_list if len(_) <= filter_limit]
    return cluster_res_list



def make_ip_cluster(ip_url_map, limit = 2):
    res = [[]]
    for ip in ip_url_map:
        if len(ip_url_map[ip])<=2:
            res[0].extend(ip_url_map[ip])
        else:
            res.append(ip_url_map[ip])
    if len(res[0])<=limit:
        res.pop(0)
    return res
    

