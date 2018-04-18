#!/usr/bin/evn python2.7
# -*- coding: utf-8 -*-

"""
聚类模块，包含如下功能
1. K-means 聚类
2  hierarchical 聚类
3. 基于字符串相似度聚类
"""
import time
import pandas as pd
import Levenshtein as ls
from collections import Counter
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

from logger import logger
from config import cfg
from vectorize import make_vectorize
from file_io import _load_vector_data, _init_dump_cluster_data, \
    _dump_cluster_data, _load_kmeans_hier_cluster_data, \
    _dump_kmeans_hier_cluster_data

# 并行进程数
N_JOBS = cfg.GLOBAL_N_JOBS
# 控制聚类的数量
N_CLUSTER_RATION = cfg.N_CLUSTER_RATION
# 控制字符串相似度
EDIT_DISTANCE_THRESH_LONG = cfg.EDIT_DISTANCE_THRESH_LONG
EDIT_DISTANCE_THRESH_SHORT = cfg.EDIT_DISTANCE_THRESH_SHORT
LONG_URL_THRESH = cfg.LONG_URL_THRESH
SHORT_URL_THRESH = cfg.SHORT_URL_THRESH


# calculate two url distance
def _core_distance_check(base_url, comp_url, metric):
    """
    :param base_url: base url string to compare
    :param comp_url: second url string to compare with base url string
    :param metric: method use to calculate distance
    :return: true of false whether the comp_url is similar with base url [bool]
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
        file_path=None):
    """
    :param data: url list[list] or k-means results file path [str]
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
    cluster_count = 0
    # url distance cluster
    for batch_index, batch_urls in enumerate(urls_list):
        cluster_list = []
        cluster_done = set()
        # for each sub cluster of k-mean results
        for index, url in enumerate(batch_urls):
            if index in cluster_done:
                continue
            logger.debug("----------------%d------------------" % index)
            cluster = [url]
            dis_check_list = Parallel(n_jobs=n_jobs)(delayed(_core_distance_check)(
                url, j, metric) for j in batch_urls[index + 1:] if j not in cluster_done)
            for dis_check_index, dis_check in enumerate(dis_check_list):
                if dis_check:
                    cluster.append(dis_check)
                    cluster_done.add(index + 1 + dis_check_index)
            cluster_done.add(url)
            # log result on console
            for line in cluster:
                logger.debug(str(line))
            logger.debug(
                ">>>>>>> BATCH:%d/%d\tTOTAL:%d\tDONE:%d" %
                (len(urls_list), batch_index + 1, len(batch_urls), len(cluster_done)))
            cluster_list.append(cluster)
            # dump result data
            cluster_count += 1
            _dump_cluster_data(file_path, cluster, cluster_count)
    logger.debug("string distance time cost:\t%f" % (time.time() - start_time))

    
# make fine grained cluster using string distance algorithms
def make_string_distance_cluster_opt(
        data,
        n_jobs=N_JOBS,
        metric="distance",
        file_path=None):
    """
    :param data: url list[list] or k-means results file path [str]
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
    # sorting the list to optimize multiprocess
    urls_list = sorted(urls_list, reverse = True, key = lambda x: len(x))
    # url distance cluster
    res_list = Parallel(n_jobs=n_jobs)(delayed(_core_distance_check_opt)(
        batch_index, batch_urls) for batch_index, batch_urls in enumerate(urls_list))
    # prepare the output file
    _init_dump_cluster_data(file_path)
    final_res = []
    for res in res_list:
        final_res += res
    for index, single_cluster in enumerate(final_res):
        _dump_cluster_data(file_path, single_cluster, index)

def _core_distance_check_opt(batch_index, batch_urls):
    cluster_list = []
    cluster_done = set()
    # for each sub cluster of k-mean results
    for index, url in enumerate(batch_urls):
        if index in cluster_done:
            continue
        cluster = [url]
        dis_check_list = []
        for comp_url in batch_urls[index + 1:]:
            if comp_url not in cluster_done:
                dis_check_list.append(_core_distance_check( url, comp_url, "distance"))
        for dis_check_index, dis_check in enumerate(dis_check_list):
            if dis_check:
                cluster.append(dis_check)
                cluster_done.add(index + 1 + dis_check_index)
        cluster_done.add(url)
        cluster_list.append(cluster)
        if index%100 == 0:
            print "batch_index %d %d/%d" %(batch_index, index, len(batch_urls))
    return cluster_list

# make hierarchical cluster
def make_hier_cluster(
        data=None,
        output_path=None,
        cluster_num=None,
        dump=True):
    """
    :param data: vector [pd.DataFrame] or vector file path [str]
    :param output_path: [str]
    :param cluster_num: [int]
    :param dump: whether dump [bool]
    :return: cluster dict [dict]
    """
    # normalize input
    assert isinstance(data, str) or isinstance(data, pd.DataFrame)
    if isinstance(data, pd.DataFrame):
        df_vector = data
    else:
        df_vector = _load_vector_data(data)

    # calculate n cluster for hierarchical clustering
    total_data_size = len(df_vector)
    n_clusters = int(total_data_size / N_CLUSTER_RATION)
    if cluster_num is not None:
        n_clusters = cluster_num
    # log n_clusters to console
    logger.debug(
        "begin to make hierarchical cluster, total_data_size: %d n_clusters: %d" %
        (total_data_size, n_clusters))
    # exception
    if n_clusters == 0 or n_clusters == 1:
        return {0: list(df_vector.index)}
    # clustering
    clustering = AgglomerativeClustering(linkage='complete',
                                         affinity="l1",
                                         n_clusters=n_clusters,
                                         compute_full_tree=False)
    clustering.fit(df_vector.values)
    df_vector["labels"] = clustering.labels_

    # dump results dict format
    res = dict()
    for i in range(n_clusters):
        df_single_cluster = df_vector.loc[df_vector["labels"] == i]
        res[i] = list(df_single_cluster.index)
    logger.debug("hcluster done!")
    if dump:
        _dump_kmeans_hier_cluster_data(res, file_path=output_path)
    else:
        return res


# make k-means cluster
def make_kmeans_cluster(
        data=None,
        output_path=None,
        cluster_num=None,
        dump=True):
    """
    :param data: vector [pd.DataFrame] or vector file path [str]
    :param output_path: [str]
    :param cluster_num: [int]
    :param dump: whether dump [bool]
    :return: cluster dict [dict]
    """
    # normalize input
    assert isinstance(data, str) or isinstance(data, pd.DataFrame)
    if isinstance(data, pd.DataFrame):
        df_vector = data
    else:
        df_vector = _load_vector_data(data)
    # calculate n cluster for K-means clustering
    total_data_size = len(df_vector)
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
            kmeans.fit(df_vector.values)
            score = metrics.calinski_harabaz_score(
                df_vector.values, kmeans.labels_)
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
    kmeans.fit(df_vector.values)
    df_vector["labels"] = kmeans.labels_

    # dump results
    res = dict()
    for i in range(optimal_n_cluster):
        df_single = df_vector.loc[df_vector["labels"] == i]
        res[i] = list(df_single.index)
    logger.debug("k-means cluster done!")
    if dump:
        _dump_kmeans_hier_cluster_data(res, file_path=output_path)
    else:
        return res


# for mass urls clustering
def make_kmeans_cluster_mass(
        urls,
        domain=True,
        path=True,
        param=True,
        thresh=50000):
    """
    :param urls: list of urls [list]
    :param domain:
    :param path:
    :param param:
    :param thresh:
    :return: cluster list [list]
    """
    assert isinstance(urls, list)
    # perform preliminary k-means cluster
    df_vector = make_vectorize(urls, domain, path, param, dump=False)
    kmeans_res_dict = make_kmeans_cluster(
        df_vector, cluster_num=int(
            len(df_vector) / thresh), dump=False)
    logger.debug("Preliminary K-means clustering complete")
    kmeans_res_list = [kmeans_res_dict[_] for _ in kmeans_res_dict]

    # perform sub clustering for big cluster
    def _core_kmeans_mass(pre_cluster_list):
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
    kmeans_res_list, flag = _core_kmeans_mass(kmeans_res_list)
    while flag != 0:
        kmeans_res_list, flag = _core_kmeans_mass(kmeans_res_list)
    return kmeans_res_list


# for mass hierarchical cluster
def make_hier_cluster_mass(
        urls_list,
        domain=True,
        path=True,
        param=True,
        output_path=None,
        dump=True):
    """
    :param urls_list: urls_list [list]
    :param domain:
    :param path:
    :param param:
    :param output_path:
    :param dump:
    :return: cluster list [list]
    """
    hcluster_res_list = []
    for urls in urls_list:
        df_vector = make_vectorize(urls, domain, path, param, dump=False)
        hcluster_res_dict = make_hier_cluster(df_vector, dump=False)
        for k, v in hcluster_res_dict.iteritems():
            hcluster_res_list.append(v)
    # filter hierarchical cluster result
    hcluster_res_list = _cluster_filter(hcluster_res_list)
    if dump:
        hier_cluster_res_dict = dict()
        for index, v in enumerate(hcluster_res_list):
            hier_cluster_res_dict[index] = v
        _dump_kmeans_hier_cluster_data(hier_cluster_res_dict, output_path)
    else:
        return hcluster_res_list


# clean cluster result list
def _cluster_filter(cluster_res_list, filter_limit=1):
    """
    :param cluster_res_list:
    :param filter_limit:
    :return:
    """
    cluster_res_list = [_ for _ in cluster_res_list if isinstance(_, list)]
    cluster_lens = [len(_) for _ in cluster_res_list]
    logger.debug("before filter %s" % str(Counter(cluster_lens)))
    cluster_res_list = [_ for _ in cluster_res_list if len(_) >= filter_limit]
    return cluster_res_list


# make ip cluster
def make_ip_cluster(ip_url_map, limit=2):
    """
    根据IP进行聚类，将相同IP的URL聚类在一起，将没聚成簇的URL放入簇列表的第一个列表
    :param ip_url_map:
    :param limit:
    :return: list of cluster [list]
    """
    res = [[]]
    for ip in ip_url_map:
        if len(ip_url_map[ip]) <= 2:
            res[0].extend(ip_url_map[ip])
        else:
            res.append(ip_url_map[ip])
    if len(res[0]) <= limit:
        res.pop(0)
    return res
