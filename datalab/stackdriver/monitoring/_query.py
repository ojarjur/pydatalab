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

import datetime
import dateutil

import gcloud.monitoring
import pandas

from . import _query_results
from . import _utils

_FREQ_MAP = dict(TODAY='D', YESTERDAY='D', WEEK_TO_DATE='W', LAST_WEEK='W',
                 MONTH_TO_DATE='MS', LAST_MONTH='MS', QUARTER_TO_DATE='QS',
                 LAST_QUARTER='QS', YEAR_TO_DATE='AS', LAST_YEAR='AS')


class TimeInterval(object):
  """User friendly relative time intervals."""
  TODAY = 'TODAY'
  YESTERDAY = 'YESTERDAY'
  WEEK_TO_DATE = 'WEEK_TO_DATE'
  LAST_WEEK = 'LAST_WEEK'
  MONTH_TO_DATE = 'MONTH_TO_DATE'
  LAST_MONTH = 'LAST_MONTH'
  QUARTER_TO_DATE = 'QUARTER_TO_DATE'
  LAST_QUARTER = 'LAST_QUARTER'
  YEAR_TO_DATE = 'YEAR_TO_DATE'
  LAST_YEAR = 'LAST_YEAR'


class Query(gcloud.monitoring.Query):
  """Query object for retrieving metric data."""

  def __init__(self,
               metric_type=gcloud.monitoring.Query.DEFAULT_METRIC_TYPE,
               interval=None, project_id=None, context=None):
    """Initializes the core query parameters.

    Example:
      query = Query('compute.googleapis.com/instance/uptime', 'LAST_MONTH')
      query = Query(interval=TimeInterval.LAST_MONTH)

    Args:
      metric_type: The metric type name. The default value is
          "compute.googleapis.com/instance/cpu/utilization", but
          please note that this default value is provided only for
          demonstration purposes and is subject to change.
      interval: Time interval for the timeseries. For example:
        TimeInterval.TODAY. Defaults to None.
      project_id: An optional project ID or number to override the one provided
          by the context.
      context: An optional Context object to use instead of the global default.

    Raises:
      ValueError: "interval" does not have a valid value.
    """
    client = _utils.make_client(project_id, context)
    super(Query, self).__init__(client, metric_type)
    if interval is not None:
      self._start_time, self._end_time = _get_timestamps(interval)

    self._results = None

  def select_metric_type(self, metric_type):
    """Copy the query and update the metric type.

    Args:
      metric_type: The metric type name.

    Returns:
      The new query object.
    """
    new_query = self.copy()
    new_query._filter.metric_type = metric_type
    return new_query

  def select_interval(self, end_time=None, start_time=None, offset=None):
    """Copy the query and set the query time interval.

    As a convenience, you can alternatively specify the interval when you create
    the query initially. Only one of start_time and offset can be specified. If
    both are None, then the interval is a point in time containing only the
    end_time.

    The dates are parsed using the dateutil parser. When a date string is
    ambiguous:
      1. For a 3-integer date, year is assumed to be last.
      2. When there is ambiguity between month and day, day comes last.
    Given the current year is 2016, '03/02', '2016/03/02' and '03/02/16' are all
    parsed as March 2nd 2016.

    Example calls:
      query = query.select_interval('2016/05/28 5pm', '2016/05/21 5pm')
      query = query.select_interval('05/28 5pm', '05/21 5pm')  # Current year
      query = query.select_interval('8pm', '2pm') # Today
      query = query.select_interval(start_time='9:00') # Today ending now.
      query = query.select_interval(offset='4h') # Last 4 hours.
      query = query.select_interval(offset='1d 4h') # Last 1 day, 4 hours.


    Args:
      end_time: The end time as a string or Python datetime object.
        Defaults to the current time.
      start_time: The start time as a string or Python datetime object.
        Defaults to None.
      offset: The offset from the end_time. This is specified as a pandas
        offset alias as shown here:
       http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    Returns:
      The new query object.

    Raises:
      ValueError: both "start_time" and "offset" are specified.
    """
    if not(start_time is None or offset is None):
      raise ValueError('At most one of start_time and offset can be specified')
    end_time = _parse_timestamp(end_time, default_now=True)
    if offset is None:
      start_time = _parse_timestamp(start_time)
    else:
      start_time = end_time - pandas.tseries.frequencies.to_offset(offset)
    return super(Query, self).select_interval(end_time, start_time)

  def align(self, per_series_aligner, seconds=0, minutes=0, hours=0, days=0):
    """Copy the query and add temporal alignment.

    If per_series_aligner is not Aligner.ALIGN_NONE, each time
    series will contain data points only on the period boundaries.

    Example:
      query = query.align(Aligner.ALIGN_MEAN, minutes=5)

    It is also possible to specify the aligner as a literal string::
      query = query.align('ALIGN_MEAN', minutes=5)

    Args:
      per_series_aligner: The approach to be used to align individual time
        series. For example: Aligner.ALIGN_MEAN.
      seconds: The number of seconds in the alignment period.
      minutes: The number of minutes in the alignment period.
      hours: The number of hours in the alignment period.
      days: The number of days in the alignment period.

    Returns:
      The new query object.
    """
    hours = days*24 + hours
    return super(Query, self).align(per_series_aligner, seconds, minutes, hours)

  def execute(self, use_cache=True):
    """Executes the query, and populates the query results.

    Args:
      use_cache: whether to use cached results or not.
    """
    if not use_cache or self._results is None:
      self._results = _query_results.QueryResults(
          self.as_dataframe(), self.metric_type, shorten_metric_type=True)

  def results(self, use_cache=True):
    """Retrieves results for the query.

    Args:
      use_cache: whether to use cached results or not.
    Returns:
      A QueryResults object containing the results.
    """
    self.execute(use_cache)
    return self._results

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

def _get_utcnow():
  return datetime.datetime.utcnow().replace(second=0, microsecond=0)

def _get_timestamps(interval):
  """Returns the start and end datetime objects given the interval name."""
  interval = interval.upper()
  freq = _FREQ_MAP.get(interval)
  if freq is None:
    raise ValueError('"interval" does not have a valid value')

  to_offset = pandas.tseries.frequencies.to_offset
  now = _get_utcnow()
  today = now.replace(hour=0, minute=0)
  ends_now = interval.endswith('_TO_DATE') or interval == TimeInterval.TODAY
  end_time = now if ends_now else None
  curr_period_begin = (today - to_offset(freq)).to_datetime()

  if interval == TimeInterval.TODAY:
    start_time = today
  elif interval == TimeInterval.YESTERDAY:
    end_time = today
    start_time = curr_period_begin
  else:
    if ends_now:
      start_time = curr_period_begin
    else:
      end_time = curr_period_begin
      start_time = (end_time - to_offset(freq)).to_datetime()

  return start_time, end_time


def _parse_timestamp(timestamp, default_now=False):
  """Parses a string timestamp into a Python datetime object."""
  if timestamp is None:
    if default_now:
      return _get_utcnow()
    else:
      return None
  elif isinstance(timestamp, datetime.datetime):
    return timestamp
  elif isinstance(timestamp, basestring):
    return dateutil.parser.parse(timestamp)
  else:
    raise TypeError('"timestamp" must be a string or datetime object')
