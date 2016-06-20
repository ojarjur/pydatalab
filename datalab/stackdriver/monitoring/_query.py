# Copyright 2016 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied.  See the License for the specific language governing permissions and limitations under
# the License.

"""Provides access to metric data as pandas dataframes."""

from __future__ import absolute_import
from past.builtins import basestring

import gcloud.monitoring
import pandas

from . import _dataframe
from . import _query_result
from . import _utils


class Query(gcloud.monitoring.Query):
  """Query object for retrieving metric data."""

  def __init__(self,
               metric_type=gcloud.monitoring.Query.DEFAULT_METRIC_TYPE,
               end_time=None, days=0, hours=0, minutes=0,
               project_id=None, context=None):
    """Initializes the core query parameters.

    The start time (exclusive) is determined by combining the
    values of "days", "hours", and "minutes", and subtracting
    the resulting duration from "end_time".

    It is also allowed to omit the end time and duration here,
    in which case the select_interval() method must be called
    before the query is executed.

    Args:
      metric_type: The metric type(s) to query. Can be a string for a single
          metric type, or a list for one or more metrics. The default value is
          "compute.googleapis.com/instance/cpu/utilization", but please note
          that this default value is provided only for demonstration purposes
          and is subject to change.
      end_time: The end time (inclusive) of the time interval for which
          results should be returned, as a datetime object. The default
          is the start of the current minute.
      days: The number of days in the time interval.
      hours: The number of hours in the time interval.
      minutes: The number of minutes in the time interval.
      project_id: An optional project ID or number to override the one provided
          by the context.
      context: An optional Context object to use instead of the global default.

    Raises:
        ValueError: "end_time" was specified but "days", "hours", and "minutes"
            are all zero. If you really want to specify a point in time, use
            the select_interval() method.
    """
    client = _utils.make_client(project_id, context)
    if isinstance(metric_type, basestring):
      metric_type = (metric_type,)
    else:
      metric_type = tuple(metric_type)
    super(Query, self).__init__(client, metric_type,
                                end_time=end_time,
                                days=days, hours=hours, minutes=minutes)

  def __iter__(self):
    return self.iter()

  def iter(self, headers_only=False, page_size=None):
    # For iteration, create a query with a single metric type.
    single_metric_query = self.copy()
    for metric_type in self._filter.metric_type:
      single_metric_query._filter.metric_type = metric_type
      for timeseries in super(Query, single_metric_query).iter(
          headers_only, page_size):
        yield timeseries

  def as_dataframe(self, label=None, labels=None):
    return _dataframe._build_dataframe(self, label, labels)

  def execute(self):
    return _query_result.QueryResult(self)

  def labels_as_dataframe(self):
    """Returns the resource and metric metadata as a dataframe.

    Returns:
      A pandas dataframe containing the resource type and resource and metric
      labels. Each row in this dataframe corresponds to the metadata from one
      time series.
    """
    headers = [{'resource': ts.resource.__dict__, 'metric': ts.metric.__dict__}
               for ts in self.iter(headers_only=True)]
    if not headers:
      return pandas.DataFrame()
    df = pandas.io.json.json_normalize(headers)

    # Add a 2 level column header.
    df.columns = pandas.MultiIndex.from_tuples(
        [col.rsplit('.', 1) for col in df.columns])

    # Re-order the columns.
    resource_keys = gcloud.monitoring._dataframe._sorted_resource_labels(
        df['resource.labels'].columns)
    sorted_columns = [('metric', 'type'), ('resource', 'type')]
    sorted_columns += sorted(col for col in df.columns
                             if col[0] == 'metric.labels')
    sorted_columns += [('resource.labels', key) for key in resource_keys]
    df = df[sorted_columns]

    # Sort the data, and clean up index values, and NaNs.
    df = df.sort_values(sorted_columns).reset_index(drop=True).fillna('')
    return df
