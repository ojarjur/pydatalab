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

"""Converts timeseries data to pandas dataframes."""
import itertools
from gcloud.monitoring import _dataframe as gcloud_dataframe


def _build_dataframe(time_series_iterable,
                     label=None, labels=None):
  """Build a pandas dataframe out of time series.

  Args:
    time_series_iterable: An iterable (e.g., a query object) yielding time
      series.
    label: The label name to use for the dataframe header. This can be the name
      of a resource label or metric label (e.g., "instance_name"), or the
      string "resource_type".
    labels:
        A list or tuple of label names to use for the dataframe header.
        If more than one label name is provided, the resulting dataframe
        will have a multi-level column header.

        Specifying neither label or labels results in a dataframe
        with a multi-level column header including the resource type and
        all available resource and metric labels.

        Specifying both label and labels is an error.

  Returns:
    A dataframe where each column represents one time series.
  """
  import pandas   # pylint: disable=import-error

  if labels is not None:
    if label is not None:
      raise ValueError('Cannot specify both "label" and "labels".')
    elif not labels:
      raise ValueError('"labels" must be non-empty or None.')

  columns = []
  headers = []
  for time_series in time_series_iterable:
    pandas_series = pandas.Series(
      data=[point.value for point in time_series.points],
      index=[point.end_time for point in time_series.points],
    )
    columns.append(pandas_series)
    headers.append(time_series.header())

  # Implement a smart default of using all available labels.
  if label is None and labels is None:
    resource_labels = set(itertools.chain.from_iterable(
      header.resource.labels for header in headers))
    metric_labels = set(itertools.chain.from_iterable(
      header.metric.labels for header in headers))
    labels = (['metric_type', 'metric_kind', 'resource_type'] +
              gcloud_dataframe._sorted_resource_labels(resource_labels) +
              sorted(metric_labels))

  # Assemble the columns into a DataFrame.
  dataframe = pandas.DataFrame.from_records(columns).T

  # Convert the timestamp strings into a DatetimeIndex.
  dataframe.index = pandas.to_datetime(dataframe.index)

  # Build a multi-level stack of column headers. Some labels may
  # be undefined for some time series.
  levels = []
  for key in labels or [label]:
    level = [_get_timeseries_label(header, key) for header in headers]
    levels.append(level)

  # Build a column Index or MultiIndex. Do not include level names
  # in the column header if the user requested a single-level header
  # by specifying "label".
  dataframe.columns = pandas.MultiIndex.from_arrays(
    levels,
    names=labels or None)

  # Sort the rows just in case (since the API doesn't guarantee the
  # ordering), and sort the columns lexicographically.
  return dataframe.sort_index(axis=0).sort_index(axis=1)


def _get_timeseries_label(timeseries, key):
  """Extends the timeseries labels to contain metric_kind and metric_type."""
  if key == 'metric_type':
    return timeseries.metric.type
  elif key == 'metric_kind':
    return timeseries.metric_kind
  else:
    return timeseries.labels.get(key, '')


def extract_single_level(dataframe, label=None, labels=None):
  """Returns a new dataframe with a single level of column header.

  Args:
    dataframe: The pandas DataFrame object that we do the manipulation on.
    label: A single level of column header to pick.
    labels: A list of one or more levels of column header to pick.

  Returns:
    A new pandas dataframe with the same data as the input dataframe, but with
    a single level for the column header. If a single level is specified, then
    that level becomes the new single level column header. Otherwise, two or
    more levels are combined using ', ' as a separator in the order specified.

    Note: Columns are reordered to have the headers in an alphabetical order.

  Raises:
    ValueError: if both label and labels are specified.
  """
  df_single = extract_levels(dataframe, label, labels)

  if len(df_single.columns.names) > 1:
    df_single.columns = [', '.join(map(str, col))
                         for col in df_single.columns.values]
  return df_single


def extract_levels(dataframe, label=None, labels=None):
  """Returns a new dataframe with the column headers of interest for a user.

  Args:
    dataframe: The pandas DataFrame object that we do the manipulation on.
    label: A single level of column header to pick.
    labels: A list of one or more levels of column header to pick.

  Returns:
    A new pandas dataframe with the same data as the input dataframe, but with
    only the levels that the user specifies. If one or more levels specified
    by the user do not exist in the input dataframe, then they will be inserted,
    and the values will be the empty string.

    Note: Columns are reordered to have the headers in an alphabetical order.

  Raises:
    ValueError: if both label and labels are specified.
  """
  import pandas   # pylint: disable=import-error
  if label is not None and labels is not None:
    raise ValueError('Exactly one of "label" and "labels" must be specified.')

  labels = [label] if label is not None else labels
  df_mult = dataframe.copy()

  # Checking for cases where nothing needs to be done: either the labels are
  # empty, or the labels match the columns.
  if  labels is None or (labels == dataframe.columns.names):
    return df_mult

  if len(labels) == 1:
    # Extracting a single level can be done in one line.
    df_mult.columns = dataframe.columns.get_level_values(labels[0])
  else:
    # Extracting multiple levels is more complex - we convert the headers
    # into a dataframe first, and then extract the levels in order.
    df_headers = pandas.DataFrame(
        dataframe.columns.tolist(), columns=dataframe.columns.names)
    df_headers = df_headers.reindex(columns=labels).fillna('')
    df_mult.columns = pandas.MultiIndex.from_arrays(
        df_headers.T.values, names=df_headers.columns.tolist())

  return df_mult.sort_index(axis=1)
