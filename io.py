#!/usr/bin/evn python2.7
# -*- coding: utf-8 -*-

"""
IO模块
数据加载与保存
"""
import pandas as pd
from logger import logger


def dump_urls(urls, file_path, csv = True, txt = False):
    if os.path.isfile(file_path):
        os.remove(file_path)
        logger.debug("OLD DATA FIND! REMOVING\t%s" % file_path)
    try:
        if csv:
            df = pd.DataFrame({"url": urls})
            df.to_csv(file_path, index=False)
        if txt:
            with open(file_path, "w") as fd:
                for line in urls:
                    fd.write(line + "\n")
        logger.debug("URLs has been dump\t%s" % file_path)
    except Exception as e:
        logger.error("%s\tFILE DUMP ERROR %s" % (file_path, str(e)))
        sys.exit(0)
        
        
def load_urls(file_path, csv = True, txt = False):
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
        
