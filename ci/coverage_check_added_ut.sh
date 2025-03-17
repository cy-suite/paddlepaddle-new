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

set +e

SYSTEM=`uname -s`
if [ -z ${BRANCH} ]; then
    BRANCH="develop"
fi

PADDLE_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"

CURDIR=`pwd`
cd $PADDLE_ROOT

CURBRANCH=`git rev-parse --abbrev-ref HEAD`
echo $CURBRANCH
if [ `git branch | grep 'prec_added_ut'` ];then
    git branch -D 'prec_added_ut'
fi
git checkout -b prec_added_ut upstream/${BRANCH}
git branch
mkdir prec_build
cd prec_build

source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/utils.sh
echo ${PADDLE_ROOT}/ci/utils.sh
init
# cmake_base ${PYTHON_ABI:-""} >prebuild.log 2>&1
cmake_base ${PYTHON_ABI:-""}

# remove line ended with .exe to get correct deleted_ut list
ctest -N | awk -F ':' '{print $2}' | sed '/^$/d' | sed '$d' | sed 's/ //g' | sed '/\.exe$/d' | grep 'test' > $PADDLE_ROOT/br-ut
#UNITTEST_DEV.spec is used for checking changes of unittests between pr and paddle_develop in the later step
spec_path_dev=${PADDLE_ROOT}/paddle/fluid/UNITTEST_DEV.spec
ctest -N | awk -F ':' '{print $2}' | sed '/^$/d' | sed '$d' > ${spec_path_dev}
cd $PADDLE_ROOT/build
spec_path_pr=${PADDLE_ROOT}/paddle/fluid/UNITTEST_PR.spec
ctest -N | awk -F ':' '{print $2}' | sed '/^$/d' | sed '$d' > ${spec_path_pr}
ctest -N | awk -F ':' '{print $2}' | sed '/^$/d' | sed '$d' | sed 's/ //g' | sed '/\.exe$/d' | grep 'test' > $PADDLE_ROOT/pr-ut
cd $PADDLE_ROOT
grep -F -x -v -f br-ut pr-ut > $PADDLE_ROOT/added_ut

echo "=======br-ut:"
cat $PADDLE_ROOT/br-ut
echo "=======pr-ut:"
cat $PADDLE_ROOT/pr-ut

sort pr-ut |uniq -d > $PADDLE_ROOT/duplicate_ut

echo "::group::New-UT:"
cat $PADDLE_ROOT/added_ut
echo "::endgroup::"
rm -rf prec_build

rm $PADDLE_ROOT/br-ut $PADDLE_ROOT/pr-ut

git checkout -f $CURBRANCH
echo $CURBRANCH
git branch -D prec_added_ut
cd $CURDIR
