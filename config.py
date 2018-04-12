import easydict

cfg = easydict.EasyDict()


'''URL CLUSTER HYPERPARAMETERS'''
cfg.GLOBAL_N_JOBS = 8

# url vectorize params
cfg.ASCII_SIZE = 128

# k-means cluster params
cfg.N_CLUSTER_RATION = 2000

# distance metric
cfg.EDIT_DISTANCE_THRESH_LONG = 0.3
cfg.EDIT_DISTANCE_THRESH_SHORT = 0.2
cfg.LONG_URL_THRESH = 100
cfg.SHORT_URL_THRESH = 20

# regex extract params
cfg.SMALL_CLUSTER_SIZE = 2
cfg.BIG_CLUSTER_SIZE = 5

cfg.MIN_SUBSTRING_SIZE = 5
cfg.SINGLE_REGEX_SIZE = 12
cfg.TOTAL_REGEX_SIZE = 20

cfg.SINGLE_REGEX_SIZE_RATIO = 0.3
cfg.TOTAL_REGEX_SIZE_RATIO = 0.3 

cfg.BIG_CLUSTER_SAMPLE_ROUND = 10

# publish
cfg.PUBLISH_FP_THRESH = 0
cfg.PUBLISH_TP_THRESH = 1
cfg.PUBLISH_RATIO = 0.5
cfg.PUBLISH_RATIO_TP_THRESH = 10

'''URL CLUSTER FILE PATH'''
# malicious url data
cfg.URL_DOMAIN = "../data/url_domain.csv"
cfg.URL_PATH = "../data/url_path.csv"
cfg.URL_PARAM = "../data/url_param.csv"


# vecotr data
cfg.VECTOR_DOMAIN_DATA = "../data/vector_domain.csv"
cfg.VECTOR_PATH_DATA = "../data/vector_path.csv"
cfg.VECTOR_PARAM_DATA = "../data/vector_param.csv"
cfg.VECTOR_DOMAIN_PATH_PARAM_DATA = "./data/vector_domain_path_param.csv"

# coarse cluster data
cfg.KMEANS_CLUSTER_DATA = "../data/cluster_kmeams.json"

# fine cluster data
cfg.CLUSTER_DISTANCE_DATA_PATH = "../data/cluster_distance.json"
cfg.CLUSTER_DISTANCE_DATA_PATH_DEBUG = "../data/cluster_distance_debug.json"

# regex data
cfg.REGEX_DISTANCE_DATA_PATH = "../data/regex_raw.txt"  # raw regex
cfg.REGEX_DISTANCE_RESULT = "../data/regex_result.txt"  # regex with FP
cfg.REGEX_DISTANCE_PUBLISH = "../data/regex_publish.txt"  # regex publish

# whitelist url for test
cfg.TEST_DATA = "../data/sangfor/safe.csv"
