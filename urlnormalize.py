#!/usr/bin/evn python2.7
# -*- coding: utf-8 -*-
import urllib
import urlparse
import logging

logging.basicConfig(level=logging.DEBUG)


class UrlNormalize:
    '''
    url标准化
    '''

    # 罕见域名
    # 静态全局变量
    def __init__(self, url):
        '''
        :param url:URL对象
        '''
        #
        if isinstance(url, unicode):
            url = url.encode("utf-8")
        self.__url = urllib.unquote_plus(url)
        # self.__src_url = url
        # 处理无协议情况
        index = self.__url.find('://')
        if index not in range(0, 6):
            self.__url = 'http://' + self.__url
        url_res = urlparse.urlparse(self.__url)
        self.__hostname = url_res.hostname
        self.__netloc = url_res.netloc
        self.__params = url_res.params
        self.__path = url_res.path
        self.__query = url_res.query

    def __del_url_scheme(self, src_url):
        """
        删除URL协议部分
        :return:
        """
        index = src_url.find('://')
        no_scheme_url = src_url
        if index in range(0, 6):
            no_scheme_url = src_url[index + 3:]

        return no_scheme_url

    def get_quote_plus_url(self, src_url=None):
        if src_url is None:
            src_url = self.__url
        src_url = self.__del_url_scheme(src_url)
        # try:
        # 防止中文的URL如：www.avlang1.com/read-htm-tid-706824-fpage-2.html1002[1月新番][脸肿字幕组]狙われた女神天使エンゼルティアー
        quote_plus_url = urllib.quote_plus(src_url, safe=':\'/?&=()@')
        # except KeyError:
        # 	logging.error(traceback.format_exc())
        # 	quote_plus_url = None
        return quote_plus_url

    def url_is_only_domain(self):
        """
        判断URL是否只有域名
        :return:
        """
        src_url = self.__del_url_scheme(self.__url)

        if src_url.find("/") != -1:
            return False
        return True

    def get_domain_path_url(self):
        """
        拼装URL只有URL的域名+path
        :return:
        """
        if self.url_is_only_domain():
            domain_path = self.__hostname
        else:
            domain_path = urlparse.urljoin(self.__url, self.__path)
        return self.get_quote_plus_url(domain_path)

    def get_src_url(self):
        """
        获取源URL数据，去除刚刚添加的协议部分
        :return:
        """
        src_url = self.__del_url_scheme(self.__url)

        return src_url

    def get_hostname(self):
        return self.__hostname

    def get_netloc(self):
        return self.__netloc

    def get_params(self):
        return self.__params

    def get_path(self):
        return self.__path

    def get_query(self):
        return self.__query

    def get_dir_list(self):
        path_list = self.__path.split("/")
        # 去空值
        dir_list = [x for x in path_list if x]
        return dir_list

    def show(self):
        '''
        显示所有数据
        '''
        for key in self.__dict__:
            logging.info('%s : %s', key, str(self.__dict__[key]))
            
    def get_key_val(self):
        if self.__params is None:
            return dict()
        windowhref = self.__params.split("&")
        res = dict()
        for i in range(len(windowhref)):
            arr = windowhref[i].split("=")
            try:
                res[arr[0]] = arr[1]
            except:
                pass
        return res


if __name__ == '__main__':
    # help(ph_data)
    #test_url = 'https://www.baidu.com/s?ie=utf-8&f=3&rsv_bp=1&rsv_idx=1&tn=baidu&wd=python%20%' \
    #           'E7%BB%98%E5%9B%BE&oq=python%25E3%2580%2580%25E6%2597%25A0%25E5%258D%258F%25E8%25A' \
    #           'E%25AE%2520url&rsv_pq=d5bbf23500005ff8&rsv_t=b44fQKE%2BeHQ6bLemvW%2FzbIcg0C4r3R70nyg' \
    #           'Ax1pAAbu0Ff7QxB4UVsVd1ck&rqlang=cn&rsv_enter=1&inputT=1729&rsv_sug3=96&rsv_sug1=39&rsv_' \
    #           'sug7=100&rsv_sug2=0&prefixsug=python%25E3%2580%2580huitu&rsp=0&rsv_sug4=2398'
    test_url = 'http://www.information-dept.com/#USAA®%20Online'
    data = UrlNormalize(test_url)
    data.show()
