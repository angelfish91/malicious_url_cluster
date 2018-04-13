#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <assert.h>
#include "maxsubstring.h"


int match(std::string str1, int p1, std::string str2, int p2, int len){
    while (len--){
        if (str1[p1] != str2[p2]) return 0;
        p1++;
        p2++;
    }
    return 1;
}

std::vector<std::string> max_substring_of_strings(std::vector<std::string> raw_strings, int min_char_count){
    std::vector<std::string> results;
    int i, j, ans_len, ans_st, size = raw_strings.size();
    while(1){
        ans_len = -1;
        for (j = raw_strings[0].size(); ans_len == -1 && j > 0; j--)
            for (i = 0 ;ans_len == -1 && i + j - 1 < raw_strings[0].size(); i++){
                int k, flag1 = 1;
                for (k = 1; flag1 && k < size; k++){
                    int s, flag2 = 0;
                    for (s = 0; !flag2 && s+j-1 < raw_strings[k].size(); s++)
                        if (match(raw_strings[0], i, raw_strings[k], s, j))
                            flag2 = 1;
                    if (!flag2)
                        flag1 = 0;
                }
                if (flag1){
                    ans_len = j;
                    ans_st = i;
                }
            }
        if (ans_len == -1)
            return results;
        else{
            if(ans_len < min_char_count)
                break;
            std::string res = raw_strings[0].substr(ans_st, ans_len);
            results.push_back(res);
            raw_strings[0] = raw_strings[0].substr(ans_st + ans_len, raw_strings[0].size()- ans_st - ans_len);
            for(int i = 1; i<raw_strings.size(); i++){
                int st = raw_strings[i].find(res);
                raw_strings[i] = raw_strings[i].substr(st + ans_len, raw_strings[i].size()- st - ans_len);
            }
        }
    }
    return results;
}


int main(){
   return 0;
}