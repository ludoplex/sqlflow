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

from runtime import db
from runtime.model import EstimatorType
from runtime.pai import table_ops


def create_predict_result_table(datasource, select, result_table, label_column,
                                train_label_column, model_type):
    """Create predict result table with given name and label column

    Args:
        datasource: current datasource
        select: sql statement to get prediction data set
        result_table: the table name to save result
        label_column: name of the label column, if not exist in select
            result, we will add a int column in the result table
        train_label_column: name of the label column when training
        model_type: type of model defined in runtime.model.oss
    """
    conn = db.connect_with_data_source(datasource)
    conn.execute(f"DROP TABLE IF EXISTS {result_table}")
    # PAI ml will create result table itself
    if model_type == EstimatorType.PAIML:
        return

    create_table_sql = (
        f"CREATE TABLE {result_table} AS SELECT * FROM {select} LIMIT 0"
    )
    conn.execute(create_table_sql)

    # if label is not in data table, add a int column for it
    schema = db.get_table_schema(conn, result_table)
    col_type = next(
        (
            ctype
            for name, ctype in schema
            if name in [train_label_column, label_column]
        ),
        "INT",
    )
    col_names = [col[0] for col in schema]
    if label_column not in col_names:
        conn.execute(conn, f"ALTER TABLE {result_table} ADD {label_column} {col_type}")
    if train_label_column != label_column and train_label_column in col_names:
        conn.execute(
            conn,
            f"ALTER TABLE {result_table} DROP COLUMN {train_label_column}",
        )


def get_create_shap_result_sql(conn, data_table, result_table, label_column):
    """Get a sql statement which create a result table for SHAP

    Args:
        conn: a database connection
        data_table: table name to read data from
        result_table: result table name
        label_column: column name of label

    Returns:
        a sql statement to create SHAP result table
    """
    schema = db.get_table_schema(conn, data_table)
    fields = [f"{f[0]} STRING" for f in schema if f[0] != label_column]
    return f'CREATE TABLE IF NOT EXISTS {result_table} ({",".join(fields)})'


def create_evaluate_result_table(datasource, result_table, metrics):
    """Create a table to hold the evaluation result

    Args:
        datasource: current datasource
        result_table: the table name to save result
        metrics: list of evaluation metrics names
    """
    table_ops.drop_tables([result_table], datasource)
    # Always add loss
    ext_metrics = ["loss"]
    if isinstance(metrics, list):
        ext_metrics.extend(metrics)
    fields = [f"{m} STRING" for m in ext_metrics]
    sql = f'CREATE TABLE IF NOT EXISTS {result_table} ({",".join(fields)});'
    conn = db.connect_with_data_source(datasource)
    conn.execute(sql)
