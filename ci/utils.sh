# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#sot

function init() {
    RED='\033[0;31m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NONE='\033[0m'

    export PADDLE_ROOT
    if [ -z "${SCRIPT_NAME}" ]; then
        SCRIPT_NAME=$0
    fi

    ENABLE_MAKE_CLEAN=${ENABLE_MAKE_CLEAN:-ON}

    # NOTE(chenweihang): For easy debugging, CI displays the C++ error stacktrace by default
    export FLAGS_call_stack_level=2
}

function build_size() {
    cat <<EOF
    ============================================
    Calculate /paddle/build size and PR whl size
    ============================================
EOF
    if [ "$1" == "paddle_inference" ]; then
        cd ${PADDLE_ROOT}/build
        cp -r paddle_inference_install_dir paddle_inference
        tar -czf paddle_inference.tgz paddle_inference
        buildSize=$(du -h --max-depth=0 ${PADDLE_ROOT}/build/paddle_inference.tgz |awk '{print $1}')
        soLibSize=$(du -h --max-depth=0 ${PADDLE_ROOT}/build/paddle_inference_install_dir/paddle/lib/libpaddle_inference.so |awk '{print $1}')
        echo "Paddle_Inference Size: $buildSize"
        echo "Paddle_Inference Dynamic Library Size: $soLibSize"
        echo "ipipe_log_param_Paddle_Inference_Size: $buildSize" >> ${PADDLE_ROOT}/build/build_summary.txt
        echo "ipipe_log_param_Paddle_Inference_So_Size: $soLibSize" >> ${PADDLE_ROOT}/build/build_summary.txt
    elif [ "$1" == "paddle_inference_c" ]; then
        cd ${PADDLE_ROOT}/build
        cp -r paddle_inference_c_install_dir paddle_inference_c
        tar -czf paddle_inference_c.tgz paddle_inference_c
        buildSize=$(du -h --max-depth=0 ${PADDLE_ROOT}/build/paddle_inference_c.tgz |awk '{print $1}')
        echo "Paddle_Inference Capi Size: $buildSize"
        echo "ipipe_log_param_Paddle_Inference_capi_Size: $buildSize" >> ${PADDLE_ROOT}/build/build_summary.txt
    else
        SYSTEM=`uname -s`
        if [ "$SYSTEM" == "Darwin" ]; then
            com='du -h -d 0'
        else
            com='du -h --max-depth=0'
        fi
        buildSize=$($com ${PADDLE_ROOT}/build |awk '{print $1}')
        echo "Build Size: $buildSize"
        echo "ipipe_log_param_Build_Size: $buildSize" >> ${PADDLE_ROOT}/build/build_summary.txt
        if ls ${PADDLE_ROOT}/build/python/dist/*whl >/dev/null 2>&1; then
            PR_whlSize=$($com ${PADDLE_ROOT}/build/python/dist |awk '{print $1}')
        elif ls ${PADDLE_ROOT}/dist/*whl >/dev/null 2>&1; then
            PR_whlSize=$($com ${PADDLE_ROOT}/dist |awk '{print $1}')
        fi
        echo "PR whl Size: $PR_whlSize"
        echo "ipipe_log_param_PR_whl_Size: $PR_whlSize" >> ${PADDLE_ROOT}/build/build_summary.txt
        PR_soSize=$($com ${PADDLE_ROOT}/build/python/paddle/base/libpaddle.so |awk '{print $1}')
        echo "PR so Size: $PR_soSize"
        echo "ipipe_log_param_PR_so_Size: $PR_soSize" >> ${PADDLE_ROOT}/build/build_summary.txt
    fi
}

function collect_ccache_hits() {
    ccache -s
    ccache_version=$(ccache -V | grep "ccache version" | awk '{print $3}')
    echo "$ccache_version"
    if [[ $ccache_version == 4* ]] ; then
        rate=$(ccache -s | grep "Hits" | awk 'NR==1 {print $5}' | cut -d '(' -f2 | cut -d ')' -f1)
        echo "ccache hit rate: ${rate}%"
    else
        rate=$(ccache -s | grep 'cache hit rate' | awk '{print $4}')
        echo "ccache hit rate: ${rate}"
    fi

    echo "ipipe_log_param_Ccache_Hit_Rate: ${rate}%" >> ${PADDLE_ROOT}/build/build_summary.txt
}

#py3

failed_test_lists=''
tmp_dir=`mktemp -d`

function get_quickly_disable_ut() {
    echo "::group::Installing httpx..."
    python -m pip install httpx
    echo "::endgroup::"
    if disable_ut_quickly=$(python ${PADDLE_ROOT}/tools/get_quick_disable_lt.py); then
        echo "========================================="
        echo "The following unittests have been disabled:"
        echo ${disable_ut_quickly}
        echo "========================================="
    else

        exit 102
        disable_ut_quickly='disable_ut'
    fi
}

function get_precision_ut_mac() {
    on_precision=0
    UT_list=$(ctest -N | awk -F ': ' '{print $2}' | sed '/^$/d' | sed '$d')
    precision_cases=""
    if [ ${PRECISION_TEST:-OFF} == "ON" ]; then
        python $PADDLE_ROOT/tools/get_pr_ut.py
        if [[ -f "ut_list" ]]; then
            echo "PREC length: "`wc -l ut_list`
            precision_cases=`cat ut_list`
        fi
    fi
    if [ ${PRECISION_TEST:-OFF} == "ON" ] && [[ "$precision_cases" != "" ]];then
        UT_list_re=''
        on_precision=1
        re=$(cat ut_list|awk -F ' ' '{print }' | awk 'BEGIN{ all_str=""}{if (all_str==""){all_str=$1}else{all_str=all_str"$|^"$1}} END{print "^"all_str"$"}')
        UT_list_prec_1='ut_list_prec2'
        for ut_case in $UT_list; do
            flag=$(echo $ut_case|grep -oE $re)
            if [ -n "$flag" ];then
                if [ -z "$UT_list_prec" ];then
                    UT_list_prec="^$ut_case$"
                elif [[ "${#UT_list_prec}" -gt 10000 ]];then
                    UT_list_prec_1="$UT_list_prec_1|^$ut_case$"
                else
                    UT_list_prec="$UT_list_prec|^$ut_case$"
                fi
            else
                echo ${ut_case} "won't run in PRECISION_TEST mode."
            fi
        done
    fi
}

function collect_failed_tests() {
    for file in `ls $tmp_dir`; do
        exit_code=0
        grep -q 'The following tests FAILED:' $tmp_dir/$file||exit_code=$?
        if [ $exit_code -ne 0 ]; then
            failuretest=''
        else
            failuretest=`grep -A 10000 'The following tests FAILED:' $tmp_dir/$file | sed 's/The following tests FAILED://g'|sed '/^$/d'`
            failed_test_lists="${failed_test_lists}
            ${failuretest}"
        fi
    done
}

function show_ut_retry_result() {
    if [ "$SYSTEM" == "Darwin" ]; then
        exec_retry_threshold_count=10
    else
        exec_retry_threshold_count=80
    fi
    if [[ "$is_retry_execute" != "0" ]]  && [[ "${exec_times}" == "0" ]] ;then
        failed_test_lists_ult=`echo "${failed_test_lists}" | grep -Po '[^ ].*$'`
        echo "========================================="
        echo "There are more than ${exec_retry_threshold_count} failed unit tests in parallel test, so no unit test retry!!!"
        echo "========================================="
        echo "The following tests FAILED: "
        echo "${failed_test_lists_ult}"
        exit 8;
    elif [[ "$is_retry_execute" != "0" ]] && [[ "${exec_times}" == "1" ]];then
        failed_test_lists_ult=`echo "${failed_test_lists}" | grep -Po '[^ ].*$'`
        echo "========================================="
        echo "There are more than 10 failed unit tests, so no unit test retry!!!"
        echo "========================================="
        echo "The following tests FAILED: "
        echo "${failed_test_lists_ult}"
        exit 8;
    else
        retry_unittests_ut_name=$(echo "$retry_unittests_record" | grep -oEi "\-.+\(" | sed 's/(//' | sed 's/- //' )
        if [ "$SYSTEM" == "Darwin" ]; then
            retry_unittests_record_judge=$(echo ${retry_unittests_ut_name}| tr ' ' '\n' | sort | uniq -c | awk '{if ($1 >=3) {print $2}}')
        else
            retry_unittests_record_judge=$(echo ${retry_unittests_ut_name}| tr ' ' '\n' | sort | uniq -c | awk '{if ($1 >=2) {print $2}}')
        fi
        if [ -z "${retry_unittests_record_judge}" ];then
            echo "========================================"
            echo "There are failed tests, which have been successful after re-run:"
            echo "========================================"
            echo "The following tests have been re-ran:"
            echo "${retry_unittests_record}"
        else
            failed_ut_re=$(echo "${retry_unittests_record_judge}" | awk BEGIN{RS=EOF}'{gsub(/\n/,"|");print}')
            echo -e "${RED}========================================${NONE}"
            echo -e "${RED}There are failed tests, which have been executed re-run,but success rate is less than 50%:${NONE}"
            echo -e "${RED}Summary Failed Tests... ${NONE}"
            echo -e "${RED}========================================${NONE}"
            echo -e "${RED}The following tests FAILED: ${NONE}"
            echo "${retry_unittests_record}" | sort -u | grep -E "$failed_ut_re" | while IFS= read -r line; do
                echo -e "${RED}${line}${NONE}"
            done
            exit 8;
        fi
    fi
}

function clean_build_files() {
    clean_files=("paddle/fluid/pybind/libpaddle.so" "third_party/flashattn/src/extern_flashattn-build/libflashattn.so" "third_party/install/flashattn/lib/libflashattn.so")

    for file in "${clean_files[@]}"; do
      file=`echo "${PADDLE_ROOT}/build/${file}"`
      if [ -f "$file" ]; then
          rm -rf "$file"
      fi
    done
}

function determine_kunlun_runner() {
    runner_name=$1

    if [[ $runner_name == "paddle-1" ]]; then
        echo "CUDA_VISIBLE_DEVICES=0,1" >> $GITHUB_ENV
        echo "XPU_CODE_1=/dev/xpu0" >> $GITHUB_ENV
        echo "XPU_CODE_2=/dev/xpu1" >> $GITHUB_ENV
    elif [[ $runner_name == "paddle-2" ]]; then
        echo "CUDA_VISIBLE_DEVICES=2,3" >> $GITHUB_ENV
        echo "XPU_CODE_1=/dev/xpu2" >> $GITHUB_ENV
        echo "XPU_CODE_2=/dev/xpu3" >> $GITHUB_ENV
    elif [[ $runner_name == "paddle-3" ]]; then
        echo "CUDA_VISIBLE_DEVICES=4,5" >> $GITHUB_ENV
        echo "XPU_CODE_1=/dev/xpu4" >> $GITHUB_ENV
        echo "XPU_CODE_2=/dev/xpu5" >> $GITHUB_ENV
    elif [[ $runner_name == "paddle-4" ]]; then
        echo "CUDA_VISIBLE_DEVICES=6,7" >> $GITHUB_ENV
        echo "XPU_CODE_1=/dev/xpu6" >> $GITHUB_ENV
        echo "XPU_CODE_2=/dev/xpu7" >> $GITHUB_ENV
    else
        echo "Unknown runner name: $runner_name"
        exit 1
    fi
    cd $GITHUB_WORKSPACE
    rm -rf * .[^.]* .??*
}

function case_count(){
    cat <<EOF
    ============================================
    Generating TestCases Count ...
    ============================================
EOF
    testcases=$1
    num=$(echo $testcases|grep -o '\^'|wc -l)
    if (( $2 == -1 )); then
        echo "exclusive TestCases count is $num"
        echo "ipipe_log_param_Exclusive_TestCases_Count: $num" >> ${PADDLE_ROOT}/build/build_summary.txt
    else
        echo "$2 card TestCases count is $num"
        echo "ipipe_log_param_${2}_Cards_TestCases_Count: $num" >> ${PADDLE_ROOT}/build/build_summary.txt
    fi
}

EXIT_CODE=0
function caught_error() {
 for job in `jobs -p`; do
        # echo "PID => ${job}"
        if ! wait ${job} ; then
            echo "At least one test failed with exit code => $?"
            EXIT_CODE=1
        fi
    done
}

function card_test() {
    set -m
    case_count $1 $2
    ut_startTime_s=`date +%s`

    testcases=$1
    cardnumber=$2
    parallel_level_base=${CTEST_PARALLEL_LEVEL:-1}

    # run ut based on the label
    if [[ "${UT_RUN_TYPE_SETTING}" == "INFER" ]];then
        run_label_mode="-L (RUN_TYPE=INFER)"
    elif [[ "${UT_RUN_TYPE_SETTING}" == "DIST" ]];then
        run_label_mode="-L (RUN_TYPE=DIST|RUN_TYPE=EXCLUSIVE)"
    elif [[ "${UT_RUN_TYPE_SETTING}" == "WITHOUT_INFER" ]];then
        run_label_mode="-LE (RUN_TYPE=INFER)"
    elif [[ "${UT_RUN_TYPE_SETTING}" == "WITHOUT_HYBRID" ]];then
        run_label_mode="-LE (RUN_TYPE=HYBRID)"
    elif [[ "${UT_RUN_TYPE_SETTING}" == "OTHER" ]];then
        run_label_mode="-LE (RUN_TYPE=INFER|RUN_TYPE=DIST|RUN_TYPE=EXCLUSIVE)"
    fi

    # get the CUDA device count, XPU device count is one
    if [ "${WITH_XPU}" == "ON" ];then
        CUDA_DEVICE_COUNT=1
    elif [ "${WITH_ROCM}" == "ON" ];then
        CUDA_DEVICE_COUNT=$(rocm-smi -i | grep DCU | wc -l)
    elif [ "${WITH_IPU}" == "ON" ];then
        CUDA_DEVICE_COUNT=1
    else
        CUDA_DEVICE_COUNT=$(nvidia-smi -L | wc -l)
    fi

    if (( $cardnumber == -1 ));then
        cardnumber=$CUDA_DEVICE_COUNT
    fi

    if (( $# > 2 )); then
        parallel_job=`expr $3 \* $parallel_level_base`
    else
        parallel_job=$parallel_level_base
    fi

    if [[ "$testcases" == "" ]]; then
        return 0
    fi

    trap 'caught_error' CHLD
    tmpfile_rand=`date +%s%N`
    NUM_PROC=$[CUDA_DEVICE_COUNT/$cardnumber]
    echo "****************************************************************"
    echo "***These unittests run $parallel_job job each time with $cardnumber GPU***"
    echo "****************************************************************"
    for (( i = 0; i < $NUM_PROC; i++ )); do
        # CUDA_VISIBLE_DEVICES http://acceleware.com/blog/cudavisibledevices-masking-gpus
        # ctest -I https://cmake.org/cmake/help/v3.0/manual/ctest.1.html?highlight=ctest
        cuda_list=()
        for (( j = 0; j < cardnumber; j++ )); do
            if [ $j -eq 0 ]; then
                    cuda_list=("$[i*cardnumber]")
                else
                    cuda_list="$cuda_list,$[i*cardnumber+j]"
            fi
        done
        tmpfile=$tmp_dir/$tmpfile_rand"_"$i
        if [ ${TESTING_DEBUG_MODE:-OFF} == "ON" ] ; then
            if [[ $cardnumber == $CUDA_DEVICE_COUNT ]]; then
                (ctest -I $i,,$NUM_PROC -R "($testcases)" -E "($disable_ut_quickly)" ${run_label_mode} -V --timeout 120 -j $parallel_job | tee $tmpfile; test ${PIPESTATUS[0]} -eq 0) &
            else
                if [ "$WITH_ROCM" == "ON" ];then
                    (env HIP_VISIBLE_DEVICES=$cuda_list ctest -I $i,,$NUM_PROC -R "($testcases)" -E "($disable_ut_quickly)" ${run_label_mode} --timeout 120 -V -j $parallel_job | tee $tmpfile; test ${PIPESTATUS[0]} -eq 0) &
                else
                    (env CUDA_VISIBLE_DEVICES=$cuda_list ctest -I $i,,$NUM_PROC -R "($testcases)" -E "($disable_ut_quickly)" ${run_label_mode} --timeout 120 -V -j $parallel_job | tee $tmpfile; test ${PIPESTATUS[0]} -eq 0) &
                fi
            fi
        else
            if [[ $cardnumber == $CUDA_DEVICE_COUNT ]]; then
                (ctest -I $i,,$NUM_PROC -R "($testcases)" -E "($disable_ut_quickly)" ${run_label_mode} --timeout 120 --output-on-failure  -j $parallel_job | tee $tmpfile; test ${PIPESTATUS[0]} -eq 0) &
            else
                if [ "$WITH_ROCM" == "ON" ];then
                    (env HIP_VISIBLE_DEVICES=$cuda_list ctest -I $i,,$NUM_PROC -R "($testcases)" -E "($disable_ut_quickly)" ${run_label_mode} --timeout 120 --output-on-failure  -j $parallel_job | tee $tmpfile; test ${PIPESTATUS[0]} -eq 0) &
                else
                    (env CUDA_VISIBLE_DEVICES=$cuda_list ctest -I $i,,$NUM_PROC -R "($testcases)" -E "($disable_ut_quickly)" ${run_label_mode} --timeout 120 --output-on-failure  -j $parallel_job | tee $tmpfile; test ${PIPESTATUS[0]} -eq 0) &
                fi

            fi
        fi
    done
    wait; # wait for all subshells to finish
    trap - CHLD
    ut_endTime_s=`date +%s`
    if (( $2 == -1 )); then
        echo "exclusive TestCases Total Time: $[ $ut_endTime_s - $ut_startTime_s ]s"
    else
        echo "$2 card TestCases Total Time: $[ $ut_endTime_s - $ut_startTime_s ]s"
    fi
    echo "$2 card TestCases finished!!!! "
    set +m
}

function check_excode() {
    if [[ $EXCODE -eq 0 ]];then
        echo "Congratulations!  Your PR passed the paddle-build."
    elif [[ $EXCODE -eq 4 ]];then
        echo "Sorry, your code style check failed."
    elif [[ $EXCODE -eq 5 ]];then
        echo "Sorry, API's example code check failed."
    elif [[ $EXCODE -eq 6 ]];then
        echo "Sorry, your pr need to be approved."
    elif [[ $EXCODE -eq 7 ]];then
        echo "Sorry, build failed."
    elif [[ $EXCODE -eq 8 ]];then
        echo "Sorry, some tests failed."
    elif [[ $EXCODE -eq 9 ]];then
        echo "Sorry, coverage check failed."
    fi
    exit $EXCODE
}
