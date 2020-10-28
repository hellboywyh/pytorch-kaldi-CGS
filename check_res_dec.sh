###
 # @Description: 
 # @version: 
 # @Author: Wang Yanhong
 # @email: 284520535@qq.com
 # @Date: 2020-10-27 00:30:54
 # @LastEditors: Wang Yanhong
 # @LastEditTime: 2020-10-27 01:04:15
### 
# !/bin/bash
# for x in $1; do [ -d $x ] && echo $x | grep "${1:-.*}" >/dev/null && grep WER $x/wer_* 2>/dev/null | ./best_wer.sh; done
# for x in $1; do [ -d $x ] && echo $x | grep "${1:-.*}" >/dev/null && grep Sum $x/*score_*/*.sys 2>/dev/null | ./best_wer.sh; done
# exit 0

# [ -d $x ] && echo $x  #如果x目录存在，则输出

for x in $1; do [ -d $x ] && echo $x | grep "${1:-.*}" >/dev/null && grep Sum $x/*score_*/*.sys 2>/dev/null | ./best_wer.sh;done

