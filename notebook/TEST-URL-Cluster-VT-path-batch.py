import os
import sys
import logging
import numpy as np
import pandas as pd
import urlparse


os.sys.path.append("/home/sparrow/sparrow/malicious_url_cluster")
from logger import logger 
from config import cfg

cfg.N_CLUSTER_RATION = 200

cfg.EDIT_DISTANCE_THRESH_LONG = 0.3
cfg.EDIT_DISTANCE_THRESH_SHORT = 0.3
cfg.LONG_URL_THRESH = 100
cfg.SHORT_URL_THRESH = 20

cfg.MIN_SUBSTRING_SIZE = 12
cfg.SINGLE_REGEX_SIZE = 12
cfg.TOTAL_REGEX_SIZE = 12
cfg.SINGLE_REGEX_SIZE_RATIO = 0.2
cfg.TOTAL_REGEX_SIZE_RATIO = 0.2


from preprocess import data_loader, url_map_ip_analysis
from vectorize import make_vectorize
from cluster_engine import make_ip_cluster, make_string_distance_cluster
from regex_engine import regex_extract, regex_check, regex_publish
import preprocess
import cluster_engine
import regex_engine


if __name__ == "__main__":
    cluster_engine.make_string_distance_cluster(data = "../../data/EXP_VirusTotal3/cluster_hier.json", 
                             metric = "distance",  
                             file_path = "../../data/EXP_VirusTotal3/cluster_distance_path.json")
    
    regex_extract(input_file_path = "../../data/EXP_VirusTotal3/cluster_distance_path.json", 
                  output_file_path = "../../data/EXP_VirusTotal3/regex_raw_path.txt")
    








