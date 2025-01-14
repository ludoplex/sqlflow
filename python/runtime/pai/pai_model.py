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

import subprocess

from runtime.dbapi.maxcompute import MaxComputeConnection
from runtime.model import oss
from runtime.model.model import Model


def get_oss_model_url(model_full_path):
    """Get OSS model save url

    Args:
        model_full_path: string, the path in OSS bucket

    Returns:
        The OSS url of the model
    """
    return f"oss://{oss.SQLFLOW_MODELS_BUCKET}/{model_full_path}"


def drop_pai_model(datasource, model_name):
    """Drop PAI model

    Args:
        datasource: current datasource
        model_name: name of the model to drop
    """
    user, passwd, address, database = MaxComputeConnection.get_uri_parts(
        datasource)
    cmd = f"drop offlinemodel if exists {model_name}"
    subprocess.run([
        "odpscmd", "-u", user, "-p", passwd, "--project", database,
        "--endpoint", address, "-e", cmd
    ],
                   check=True)


def get_oss_model_save_path(datasource, model_name, user=""):
    if not model_name:
        return None
    _, _, _, project = MaxComputeConnection.get_uri_parts(datasource)
    if user == "":
        user = "unknown"
    return "/".join([project, user, model_name])


def clean_oss_model_path(oss_path):
    bucket = oss.get_models_bucket()
    oss.delete_oss_dir_recursive(bucket, oss_path)


def get_saved_model_type_and_estimator(datasource, model_name):
    """Get oss model type and estimator name, model can be:
    1. PAI ML models: model is saved by pai
    2. xgboost: on OSS with model file xgboost_model_desc
    3. PAI tensorflow models: on OSS with meta file: tensorflow_model_desc

    Args:
        datasource: the DBMS connection URI.
        model_name: the model to get info

    Returns:
        If model is TensorFlow model, return type and estimator name
        If model is XGBoost, or other PAI model, just return model type
    """
    # FIXME(typhoonzero): if the model not exist on OSS, assume it's a random
    # forest model should use a general method to fetch the model and see the
    # model type.
    meta = Model.load_metadata_from_db(datasource, model_name)
    return meta.get_type(), meta.get_meta("class_name")
