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

# NOTE: ALPS supports tensorflow 1.15 currently, should run this example with
# TensorFlow 1.15.x installed.

import os
import shutil

import tensorflow as tf
# pylint: disable=E0401
# need to import GroupedSparseColumn, SparseColumn when it's used
from alps.framework.column.column import DenseColumn
from alps.framework.experiment import EstimatorBuilder
from alps.io.base import OdpsConf
# pylint: enable=E0401
from runtime.alps.train import train
from runtime.tensorflow.get_tf_version import tf_is_version2


class SQLFlowEstimatorBuilder(EstimatorBuilder):
    def _build(self, experiment, run_config):
        feature_columns = [
            tf.feature_column.numeric_column(col_name)
            for col_name in [
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
            ]
        ]
        return tf.estimator.DNNClassifier(  # pylint: disable=no-member
            n_classes=3,
            hidden_units=[10, 20],
            config=run_config,
            feature_columns=feature_columns)


if __name__ == "__main__":
    if tf_is_version2():
        raise ValueError("ALPS must run with TensorFlow == 1.15.x")
    odps_project = os.getenv("SQLFLOW_TEST_DB_MAXCOMPUTE_PROJECT")
    odps_conf = OdpsConf(
        accessid=os.getenv("SQLFLOW_TEST_DB_MAXCOMPUTE_AK"),
        accesskey=os.getenv("SQLFLOW_TEST_DB_MAXCOMPUTE_SK"),
        # endpoint should looks like:
        # "https://service.cn.maxcompute.aliyun.com/api"
        endpoint=os.getenv("SQLFLOW_TEST_DB_MAXCOMPUTE_ENDPOINT"),
        project=odps_project)

    features = [
        DenseColumn(name=col_name, shape=[1], dtype="float32")
        for col_name in [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]
    ]
    labels = DenseColumn(name="class", shape=[1], dtype="int", separator=",")

    try:
        os.mkdir("scratch")
    except FileExistsError:
        pass

    train(
        SQLFlowEstimatorBuilder(),
        odps_conf=odps_conf,
        project=odps_project,
        train_table=f"{odps_project}.sqlflow_test_iris_train",
        eval_table=f"{odps_project}.sqlflow_test_iris_test",
        features=features,
        labels=labels,
        feature_map_table="",
        feature_map_partition="",
        epochs=1,
        batch_size=2,
        shuffle=False,
        shuffle_bufsize=128,
        cache_file="",
        max_steps=1000,
        eval_steps=100,
        eval_batch_size=1,
        eval_start_delay=120,
        eval_throttle=600,
        drop_remainder=True,
        export_path="./scratch/model",
        scratch_dir="./scratch",
        user_id="",
        engine_config={"name": "LocalEngine"},
        exit_on_submit=False,
    )
    shutil.rmtree("scratch")
