import os
import sys
import logging
import numpy as np
import pandas as pd
import urlparse

os.sys.path.append("/home/sparrow/sparrow/malicious_url_cluster")
from logger import logger 
from config import cfg

cfg.EDIT_DISTANCE_THRESH_LONG = 0.3
cfg.EDIT_DISTANCE_THRESH_SHORT = 0.3
cfg.LONG_URL_THRESH = 80
cfg.SHORT_URL_THRESH = 20

cfg.MIN_SUBSTRING_SIZE = 12
cfg.SINGLE_REGEX_SIZE = 30
cfg.TOTAL_REGEX_SIZE = 30


import preprocess
import cluster_engine
import regex_engine

from logger import logger
from preprocess import url_statistic_analysis, dump_urls
from vectorize import make_vectorize
from cluster_engine import make_kmeans_cluster, make_string_distance_cluster
from regex_engine import regex_extract, regex_check, regex_publish


fh = logging.FileHandler(filename = "./log/phishtank.log", mode = 'a', delay = False)
logger.addHandler(fh)

def data_loader(filepath, csv = True, txt = False):
    try:
        if csv:
            df = pd.read_csv(filepath)
            urls = list(df.url)
        if txt:
            with open(filepath, "r") as fd:
                urls = [_.strip() for _ in fd]
        logger.debug("Malware URL Count:\t%d" %len(urls))    
        return urls
    except Exception as e:
        logger.error("%s FILE OPEN ERROR! %s" %(filepath, str(e)))
        sys.exit(0)

            
if __name__ == "__main__":
    # laod urls
    urls = data_loader("../data/phishtank/verified_online.csv")
    url_domain, url_path, url_param = preprocess.url_statistic_analysis(urls)
  
    dump_urls(url_domain, "../data/phishtank/phishtank_url_domain.csv")
    dump_urls(url_path, "../data/phishtank/phishtank_url_path.csv")
    dump_urls(url_param, "../data/phishtank/phishtank_url_param.csv")
    
    
    # URL Vectorize
    df_param = make_vectorize(url_param, 
                  domain = False, 
                  path = True, 
                  param = True, 
                  output_path = "../data/phishtank/phishtank_vector_param.csv")
    df_path = make_vectorize(url_path, 
                  domain = True, 
                  path = True, 
                  param = False, 
                  output_path = "../data/phishtank/phishtank_vector_path.csv")

    # k-means cluster
    make_kmeans_cluster(data = df_param, 
                  output_path = '../data/phishtank/phishtank_cluster_kmeams_param.json')
    make_kmeans_cluster(data = df_path, 
                  output_path = '../data/phishtank/phishtank_cluster_kmeams_path.json')
    
    # string distance cluster
    make_string_distance_cluster(data = '../data/phishtank/phishtank_cluster_kmeams_param.json', 
                        metric = "distance", 
                        file_path = "../data/phishtank/phishtank_cluster_distance_param.json")
    
    make_string_distance_cluster(data = '../data/phishtank/phishtank_cluster_kmeams_path.json', 
                        metric = "distance", 
                        file_path = "../data/phishtank/phishtank_cluster_distance_path.json")
    
    # extract regex
    regex_extract(input_file_path = "../data/phishtank/phishtank_cluster_distance_param.json", 
              output_file_path = "../data/phishtank/phishtank_regex_raw_param.txt")
    
    regex_extract(input_file_path = "../data/phishtank/phishtank_cluster_distance_path.json", 
              output_file_path = "../data/phishtank/phishtank_regex_raw_path.txt")
    
    # performance check
    regex_check(input_file_path = "../data/phishtank/phishtank_regex_raw_param.txt", 
            test_benign_file_path = "../data/sangfor/safe.csv",
            test_malicious_file_path = "../data/phishtank/phishtank_url_param.csv",
            result_file_path = "../data/phishtank/phishtank_regex_result_param.txt")
    
    regex_check(input_file_path = "../data/phishtank/phishtank_regex_raw_path.txt", 
            test_benign_file_path = "../data/sangfor/safe.csv",
            test_malicious_file_path = "../data/phishtank/phishtank_url_path.csv",
            result_file_path = "../data/phishtank/phishtank_regex_result_path.txt")
    
    # publish regex
    regex_publish(result_file_path= "../data/phishtank/phishtank_regex_result_param.txt",
              publish_file_path="../data/phishtank/phishtank_regex_publish_param.txt")
    
    regex_publish(result_file_path= "../data/phishtank/phishtank_regex_result_path.txt",
              publish_file_path="../data/phishtank/phishtank_regex_publish_path.txt")

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
