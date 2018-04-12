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
