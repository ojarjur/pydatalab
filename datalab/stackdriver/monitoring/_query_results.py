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

"""QueryResults object with visualization methods."""

from __future__ import absolute_import
from past.builtins import basestring

import operator
import re

import numpy
import pandas

from IPython import display

from . import _utils
from . import _visualization


class QueryResults(object):
  """QueryResults object contains the results of executing a query."""

  def __init__(self, dataframe, metric_type):
    self._dataframe = dataframe
    self.metric_type = metric_type

  @property
  def dataframe(self):
    return self._dataframe.copy()

  @property
  def empty(self):
    """Returns True iff the results are empty."""
    return self._dataframe.empty

  @property
  def frequency(self):
    """The frequency of datapoints in seconds."""
    if len(self._dataframe.index) < 2:
      return numpy.nan
    ts_freq = self._dataframe.index[1] - self._dataframe.index[0]
    return int(ts_freq.total_seconds())

  @property
  def timestamps(self):
    """The list of timestamps present in this result."""
    return self._dataframe.index.to_pydatetime()

  @property
  def labels(self):
    """The list of resource and metric labels in this data as dicts."""
    multiindex = self._dataframe.columns
    col_df = pandas.DataFrame(multiindex.tolist(), index=multiindex.names)
    return list(col_df.to_dict().itervalues())

  @property
  def label_keys(self):
    return self._dataframe.columns.names

  def __repr__(self):
    """Return a representation string with the points elided."""
    rep = '<QueryResults for metric=%r:' % self.metric_type
    if self.empty:
      return rep + ' empty>'
    return (
        '{rep_start}\n'
        '{num2} resources\n'
        '{num1} timestamps with 1 point every {freq} seconds>'
    ).format(
        rep_start=rep,
        num1=len(self._dataframe.index),
        num2=len(self._dataframe.columns),
        freq=self.frequency,
    )

  def table(self, max_points=-1):
    """Visualize the results as an HTML table."""
    display.display(self._dataframe.head(max_points))

  def is_compatible(self, other):
    if not isinstance(other, QueryResults):
      return False
    if self.label_keys != other.label_keys:
      return False
    if not set(self._dataframe.index) & set(other._dataframe.index):
      return False
    if not set(self._dataframe.columns) & set(other._dataframe.columns):
      return False
    return True

  def _binary_operation(self, other, operation, reverse_order=False):
    operation_to_symbol = dict(
        add='+', sub='-', mul='*', div='/', truediv='/', floordiv='//',
        pow='**', mod='%')
    operand1 = self._dataframe
    if isinstance(other, QueryResults):
      if not self.is_compatible(other):
        raise ValueError(
            'The other QueryResults is not compatible for a binary operation')
      operand2 = other._dataframe
      other_metric_type = other.metric_type
    elif isinstance(other, (int, float, long)):
      operand2 = other
      other_metric_type = other
    else:
      raise TypeError(
          '%r is not a valid input for adding to QueryResults' % other)

    if reverse_order:
      new_df = getattr(operator, operation)(operand2, operand1)
      new_metric_type = '(%s %s %s)' % (
          other_metric_type, operation_to_symbol[operation], self.metric_type)
    else:
      new_df = getattr(operator, operation)(operand1, operand2)
      new_metric_type = '(%s %s %s)' % (
          self.metric_type, operation_to_symbol[operation], other_metric_type)

    return QueryResults(new_df, new_metric_type)

  def __add__(self, other, reverse_order=False):
    return self._binary_operation(other, 'add', reverse_order)

  def __radd__(self, other):
    return self.__add__(other, True)

  def __sub__(self, other, reverse_order=False):
    return self._binary_operation(other, 'sub', reverse_order)

  def __rsub__(self, other):
    return self.__sub__(other, True)

  def __mul__(self, other, reverse_order=False):
    return self._binary_operation(other, 'mul', reverse_order)

  def __rmul__(self, other):
    return self.__mul__(other, True)

  def __div__(self, other, reverse_order=False):
    return self._binary_operation(other, 'div', reverse_order)

  def __rdiv__(self, other):
    return self.__div__(other, True)

  def __truediv__(self, other, reverse_order=False):
    return self._binary_operation(other, 'truediv', reverse_order)

  def __rtruediv__(self, other):
    return self.__truediv__(other, True)

  def __floordiv__(self, other, reverse_order=False):
    return self._binary_operation(other, 'floordiv', reverse_order)

  def __rfloordiv__(self, other):
    return self.__floordiv__(other, True)

  def __pow__(self, other, reverse_order=False):
    return self._binary_operation(other, 'pow', reverse_order)

  def __rpow__(self, other):
    return self.__pow__(other, True)

  def __mod__(self, other, reverse_order=False):
    return self._binary_operation(other, 'mod', reverse_order)

  def _unary_operation(self, operation):
    new_df = getattr(numpy, operation)(self._dataframe)
    new_metric_type = '%s(%s)' % (operation, self.metric_type)
    return QueryResults(new_df, new_metric_type)

  def __rmod__(self, other):
    return self.__mod__(other, True)

  def __abs__(self):
    return self._unary_operation('abs')

  def abs(self):
    return self._unary_operation('abs')

  def floor(self):
    return self._unary_operation('floor')

  def ceil(self):
    return self._unary_operation('ceil')

  def round(self):
    return self._unary_operation('round')

  def log10(self):
    return self._unary_operation('log10')

  def log2(self):
    return self._unary_operation('log2')

  def sqrt(self):
    return self._unary_operation('sqrt')

  def timesplit(self, freq, use_average=False):
    """Split's the result based on the specified frequency

    Args:
     freq: The frequency at which to split the query results. It borrows its
       format from pandas Offset Aliases, but is restricted to:
       [H, D, W, M, Q, A]. For 1 day, 'D'/'1D' will work fine.
     use_average: If True, it takes an average over the previous intervals, so
       the latest interval (say week) can be compared to the average of the past
       intervals.

    Returns:
      A list of QueryResults split based on the input frequency. All the results
      after the 1st one are time-shifted so as to be aligned with it.
    """
    if self.empty:
      return []

    freq = freq.upper()
    freq_to_full = dict(H='hour', D='day', W='week', M='month', Q='quarter',
                        A='year')
    regex_match = re.match(r'(\d*)(%s)$' % '|'.join(freq_to_full), freq)
    if not regex_match:
      raise ValueError('"freq" does not have a valid value')
    freq_count = 1 if not regex_match.group(1) else int(regex_match.group(1))
    freq_name = freq_to_full[regex_match.group(2)]

    freq = '%sS' % freq if freq in ['M', 'Q', 'A'] else freq
    offset = pandas.tseries.frequencies.to_offset(freq)
    one_ns = pandas.Timedelta('1ns')

    split_dataframes = []

    # Pick the first index value as the start, and zero out anything < 1 hour.
    first_start_time = self._dataframe.index[0].replace(
        minute=0, second=0, microsecond=0)
    if freq != 'H':
      first_start_time = first_start_time.replace(hour=0)
    if first_start_time != first_start_time + 0*offset:
      # If not on the boundary, move the start time back.
      first_start_time -= offset

    last_start_time = None
    new_start_time = first_start_time

    # Split the dataframe into the required intervals.
    while new_start_time <= self._dataframe.index[-1]:
      start_time = new_start_time
      new_start_time += offset
      end_time = new_start_time - one_ns
      new_df = self._dataframe[start_time: end_time]
      last_start_time = new_df.index[0]
      split_dataframes.append(new_df)

    # Time shift the dataframes to line up with the last dataframe.
    for i, df in enumerate(split_dataframes[:-1]):
      split_dataframes[i] = df.tshift(freq=last_start_time - df.index[0])

    # Create a query result from the last interval.
    if freq_count == 1:
      first_metric_type = 'Latest %s' % freq_name
    else:
      first_metric_type = 'Latest %d %ss' % (freq_count, freq_name)
    results = [QueryResults(split_dataframes[-1], first_metric_type)]

    # Use the remaining dataframes for creating one or more results.
    other_dataframes = list(reversed(split_dataframes[:-1]))

    if len(other_dataframes) > 1 and use_average:
      # Collect all the dataframes for old intervals, and take their mean.
      concat_df = pandas.concat(other_dataframes)
      aggregated_df = concat_df.groupby(level=0).mean()
      new_metric_type = 'Avg of previous %d %d-%s invervals' % (
          len(other_dataframes), freq_count, freq_name)
      results.append(QueryResults(aggregated_df, new_metric_type))
    else:
      for i, df in enumerate(other_dataframes):
        count = (i + 1) * freq_count
        unit = freq_name if count == 1 else '%ss' % freq_name
        new_metric_type = '%d %s ago' % (count, unit)
        results.append(QueryResults(df, new_metric_type))

    return results

  def delta(self):
    """Returns a new result with the change of the metric over time."""
    new_metric_type = 'delta(%s)' % self.metric_type
    return QueryResults(self._dataframe.diff(), new_metric_type)

  def rate_of_change(self):
    """Returns a new result with the rate of change over time."""
    new_df = self._dataframe.diff() / self.frequency
    new_metric_type = 'rate_of_change(%s)' % self.metric_type
    return QueryResults(new_df, new_metric_type)

  def integrate(self):
    """Returns a new result with values multiplied by frequency in seconds."""
    new_df = self._dataframe * self.frequency
    new_metric_type = 'integrate(%s)' % self.metric_type
    return QueryResults(new_df, new_metric_type)

  def aggregate(self, by, func=numpy.mean, func_name=None):
    """Aggregates the result based on a given header level.

    Args:
      by: A header level in the data over which to aggregate. E.g. zone.
      func: A function to find an aggregate over the values at a given time.
       numpy functions can be passed by name, e.g. "mean", "std", etc. Defaults
       to numpy.mean.
      func_name: Name of the function to use to assign the metric type of the
        new QueryResults - this is useful when passing in a lambda function.

    Returns:
      A new QueryResults with only the specified level in its column header, and
      with the aggregate for the specified time at each row.
    """
    if isinstance(func, basestring):
      func_name = func_name or func
      func = getattr(numpy, func)
    assert func is None or hasattr(func, '__call__')
    func_name = func_name or func.__name__
    if by is None:
      new_df = self._dataframe.apply(func, axis=1).to_frame(name='global')
      new_metric_type = '%s.%s()'% (self.metric_type, func_name)
    else:
      new_df = self._dataframe.groupby(level=by, axis=1).aggregate(func)
      new_metric_type = '%s.%s(%s)' % (self.metric_type, func_name, by)
    return QueryResults(new_df, new_metric_type)

  def mean(self, by=None):
    """Finds the mean of the result for a given header level."""
    return self.aggregate(by, numpy.mean)

  def min(self, by=None):
    """Finds the min of the result for a given header level."""
    return self.aggregate(by, numpy.min)

  def max(self, by=None):
    """Finds the max of the result for a given header level."""
    return self.aggregate(by, numpy.max)

  def sum(self, by=None):
    """Finds the sum of the result for a given header level."""
    return self.aggregate(by, numpy.sum)

  def std(self, by=None):
    """Finds the standard deviation of the result for a given header level."""
    return self.aggregate(by, numpy.std)

  def var(self, by=None):
    """Finds the variance of the result for a given header level."""
    return self.aggregate(by, numpy.var)

  def percentile(self, by=None, quantile=50):
    """Finds the percentile of the result for a given header level."""
    percentile_func = lambda x: numpy.nanpercentile(x, q=quantile)
    return self.aggregate(by, percentile_func, 'percentile_%s' % quantile)

  def _top_sorted(self, count=5, percentage=None, aggregation=numpy.mean,
                  is_top=True):
    df = self._dataframe
    if percentage is not None:
      if not 0 < percentage <= 100:
        raise ValueError('"percentage" must a number between 0 and 100')
      count = int(numpy.ceil(percentage/100.0 * len(df.columns)))

    if isinstance(aggregation, basestring):
      func, func_name = getattr(numpy, aggregation), aggregation
    else:
      func, func_name = aggregation, aggregation.__name__

    sorted_df = df.apply(func).sort_values(ascending=is_top)
    new_df = df.reindex_axis(sorted_df.index, axis=1).iloc[:, -count:]

    caller = 'top' if is_top else 'bottom'
    number = count if percentage is None else '%s%%' % percentage
    new_metric_type = '%s_%s_%s(%s)' % (
        caller, number, func_name, self.metric_type)
    return QueryResults(new_df, new_metric_type)

  def top(self, count=5, percentage=None, aggregation=numpy.mean):
    """Returns the top timeseries in the QueryResults.

    Args:
      count: The number of top results to pick. Defaults to 5.
      percentage: The percentage of top results to pick. When both percentage
        and count are specified, the percentage is used.
     aggregation: A numpy aggregation method as a string (e.g. "std", "min") or
       a function (numpy.std, numpy.min). Defaults to "mean".

    Returns:
      A new QueryResults with only the top timeseries columns.
    """
    return self._top_sorted(count, percentage, aggregation, is_top=True)

  def bottom(self, count=5, percentage=None, aggregation=numpy.mean):
    """Returns the bottom timeseries in the QueryResults.

    Args:
      count: The number of bottom results to pick. Defaults to 5.
      percentage: The percentage of bottom results to pick. When both
        percentage and count are specified, the percentage is used.
     aggregation: A numpy aggregation method as a string (e.g. "std", "min") or
       a function (numpy.std, numpy.min). Defaults to "mean".

    Returns:
      A new QueryResults with only the top timeseries columns.
    """
    return self._top_sorted(count, percentage, aggregation, is_top=False)

  def linechart(self, partition_by=None, annotate_by=None, **kwargs):
    """Draws a plotly linechart for this QueryResults.

    Args:
      partition_by: One or more labels to partition the results into separate
        linecharts. It can be a string or a list. Defaults to 'metric_type'.
      annotate_by: One or more labels to aggregate and annotate each linechart
        by. It can be a string or a list/tuple.
      **kwargs: Any arguments to pass in to the layout engine
        plotly.graph_objs.Layout().
    """
    _visualization.plot_query_results(
        self, 'linechart', partition_by, annotate_by, **kwargs)

  def heatmap(self, partition_by=None, annotate_by=None,
              zrange=None, colorscale=None, is_logscale=False,
              is_divergent=False, **kwargs):
    """Draws a plotly heatmap for this QueryResults.

    Args:
      partition_by: One or more labels to partition the results into separate
        heatmaps. It can be a string or a list. Defaults to 'metric_type'.
      annotate_by: One or more labels to aggregate and annotate each heatmap
        by. It can be a string or a list/tuple.
      zrange: A list or tuple of length 2 numbers containing the range to use
        for the colormap. If not specified, then it is calculated from the
        dataframe.
      colorscale: str, A colorscale supported by matplotlib. See:
        http://matplotlib.org/examples/color/colormaps_reference.html
      is_logscale: boolean, if True, then a logarithmic colorscale is used.
      is_divergent: boolean, specifies if the data has diverging values. If
        False, we check if the data diverges around 0, and use an appropriate
        default colormap. Ignored if you specify the colormap.
      **kwargs: Any arguments to pass in to the layout engine
        plotly.graph_objs.Layout().
    """
    _visualization.plot_query_results(
        self, 'heatmap', partition_by, annotate_by, zrange=zrange,
        colorscale=colorscale, is_logscale=is_logscale,
        is_divergent=is_divergent, **kwargs)
