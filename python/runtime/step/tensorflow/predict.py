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

import six
import tensorflow as tf
from runtime import db
from runtime.dbapi.paiio import PaiIOConnection
from runtime.feature.compile import compile_ir_feature_columns
from runtime.feature.derivation import get_ordered_field_descs
from runtime.model.model import Model
from runtime.tensorflow.get_tf_model_type import is_tf_estimator
from runtime.tensorflow.import_model import import_model
from runtime.tensorflow.load_model import pop_optimizer_and_loss
from runtime.tensorflow.predict import estimator_predict, keras_predict


def predict_step(datasource,
                 select,
                 result_table,
                 result_column_names,
                 train_label_idx,
                 model,
                 extra_result_cols=[],
                 pai_table=None):
    if isinstance(model, six.string_types):
        model = Model.load_from_db(datasource, model)
    else:
        assert isinstance(
            model, Model
        ), f"not supported model type {type(model)}"

    model_params = model.get_meta("attributes")
    train_fc_map = model.get_meta("features")
    label_meta = model.get_meta("label")
    train_label_desc = label_meta.get_field_desc()[0] if label_meta else None
    train_label_name = train_label_desc.name if train_label_desc else None
    estimator_string = model.get_meta("class_name")
    save = "model_save"

    field_descs = get_ordered_field_descs(train_fc_map)
    feature_column_names = [fd.name for fd in field_descs]
    feature_metas = dict([(fd.name, fd.to_dict(dtype_to_string=True))
                          for fd in field_descs])
    feature_columns = compile_ir_feature_columns(train_fc_map,
                                                 model.get_type())

    is_pai = bool(pai_table)
    if is_pai:
        conn = PaiIOConnection.from_table(pai_table)
        select = None
    else:
        conn = db.connect_with_data_source(datasource)

    label_name = result_column_names[-len(extra_result_cols) - 1]
    selected_cols = result_column_names[:-len(extra_result_cols) - 1]
    if train_label_idx >= 0:
        selected_cols = (
            selected_cols[:train_label_idx] + [train_label_name]
        ) + selected_cols[train_label_idx:]

    estimator = import_model(estimator_string)
    model_params.update(feature_columns)
    is_estimator = is_tf_estimator(estimator)
    predict_generator = db.db_generator(conn, select)

    pop_optimizer_and_loss(model_params)
    if not is_estimator:
        if not issubclass(estimator, tf.keras.Model):
            # functional model need field_metas parameter
            model_params["field_metas"] = feature_metas
        print("Start predicting using keras model...")
        keras_predict(estimator, model_params, save, result_table,
                      feature_column_names, feature_metas, train_label_name,
                      label_name, conn, predict_generator, selected_cols,
                      extra_result_cols)
    else:
        # TODO(sneaxiy): support extra_result_cols for estimator
        model_params['model_dir'] = save
        print("Start predicting using estimator model...")
        estimator_predict(result_table, feature_column_names, feature_metas,
                          train_label_name, label_name, conn,
                          predict_generator, selected_cols)

    print(f"Done predicting. Predict table : {result_table}")
    conn.close()
