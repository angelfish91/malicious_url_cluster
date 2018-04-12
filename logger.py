import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s\t%(message)s',
    #filename = "/home/sparrow/sparrow/malicious_url_cluster/log/url_checker.log",
    #filemode = 'a+',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger()
