#!/bin/bash
# Copyright 2020 The SQLFlow Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

changed_fileext=$(git diff --name-only HEAD..origin/develop --|awk -F. '{print $NF}'|uniq)
if [[ "$changed_fileext" == "md" ]]; then
    echo "Only Markdown files changed.  No need to run unit tests."
    exit 0
fi

export SQLFLOW_TEST_DB=mysql

python -c "import sqlflow_models"
python -c "import runtime.db"

go generate ./...
go install ./...

# FIXME(typhoonzero): Skip test using experimental codegen
# for USE_EXPERIMENTAL_CODEGEN in "true" ""; do
#     export SQLFLOW_USE_EXPERIMENTAL_CODEGEN=$USE_EXPERIMENTAL_CODEGEN
#     echo "Run Go tests when SQLFLOW_USE_EXPERIMENTAL_CODEGEN=$SQLFLOW_USE_EXPERIMENTAL_CODEGEN"
#     gotest -p 1 -covermode=count -coverprofile=coverage.txt -timeout 1500s  -v ./...
# done
gotest -p 1 -covermode=count -coverprofile=coverage.txt -timeout 1800s  -v ./... -run TestEnd2EndMySQL


# When running the following command, the TensorFlow FLAGS module would pass
# ["discover", "-v", "python", "*_test.py"] as the sys.argv to init the
# tf.app.flags.FLAGS, which would cause error. So we use the
# SQLFLOW_USE_DEFAULT_FLAGS to avoid the parse error.
export SQLFLOW_USE_DEFAULT_FLAGS=true
coverage run -m unittest discover -v python "*_test.py"
