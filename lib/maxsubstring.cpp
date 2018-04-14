#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <assert.h>
#include <algorithm>
#include "maxsubstring.h"


int match(std::string str1, int p1, std::string str2, int p2, int len){
    while (len--){
        if (str1[p1] != str2[p2]) return 0;
        p1++;
        p2++;
    }
    return 1;
}


struct fres{
    std::string res;
    int index;
};

bool comp(const fres &a, const fres &b){
    return a.index < b.index;
}

std::vector<std::string> max_substring_of_strings(std::vector<std::string> raw_strings, int min_char_count){
    std::vector<fres> results;
    std::vector<std::string> fresults;
    int i, j, ans_len, ans_st, size = raw_strings.size();
    std::string fstr = raw_strings[0];
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
            break;
        else{
            if(ans_len < min_char_count)
                break;
            // store the results
            std::string substring = raw_strings[0].substr(ans_st, ans_len);
            fres temp_fres;
            temp_fres.res = substring;
            temp_fres.index = fstr.find(substring);
            results.push_back(temp_fres);
            // reconstruct string array to run next round
            raw_strings[0] = raw_strings[0].substr(0, ans_st) + "\t" + 
                raw_strings[0].substr(ans_st + ans_len, raw_strings[0].size()- ans_st - ans_len);
            for(int i = 1; i<raw_strings.size(); i++){
                int st = raw_strings[i].find(substring);
                raw_strings[i] = raw_strings[i].substr(0, st) + "\n" + 
                    raw_strings[i].substr(st + ans_len, raw_strings[i].size()- st - ans_len);
            }
        }
    }
    // sort result 
    sort(results.begin(),results.end(),comp);
    for(int i=0; i< results.size(); i++){
        fresults.push_back(results[i].res);
    }
    return fresults;
}




int main(){
   return 0;
}