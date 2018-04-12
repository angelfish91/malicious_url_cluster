cimport cython
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "maxsubstring.h":
    extern vector[string] max_substring_of_strings(vector[string] raw_strings, int min_char_count)

def maxsubstring(in_list, thresh = 4):
    assert type(in_list) == list
    assert type(thresh) == int
    cdef vector[string] input_string
    for i in in_list:
        input_string.push_back(i)
        
    out = max_substring_of_strings(input_string, thresh)
    return out