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

import pandas as pd
import numpy as np

from . import _visualization


class Aggregator(object):
  COUNT = 'count'
  MIN = 'min'
  MAX = 'max'
  MEAN = 'mean'
  MEDIAN = 'median'
  STDDEV = 'std'
  SUM = 'sum'
  VARIANCE = 'var'


class QueryResults(object):
  """QueryResults object contains the results of executing a query."""
  _DEFAULT_PARTITION_BY = 'metric_type'
  _ALL_PLOT_KINDS = ('linechart', 'heatmap')

  def __init__(self, query, use_short_metric_types=True):
    self._query = query
    self._dataframe = query.as_dataframe()
    if use_short_metric_types:
      self._shorten_metric_types()

  @property
  def empty(self):
    return self._dataframe.empty

  @property
  def query(self):
    """The underlying query that generated these results."""
    return self._query

  def _shorten_metric_types(self):
    """Shorten the metric types to only contain the value after the last '/'."""
    new_columns = [[col[0].split('/')[-1]] + list(col[1:])
                   for col in self._dataframe.columns]
    self._dataframe.columns = pd.MultiIndex.from_tuples(
        new_columns, names=self._dataframe.columns.names)

  def plot(self, kind='linechart', partition_by=_DEFAULT_PARTITION_BY,
           annotate_by=None, aggregation_method=Aggregator.MEAN, **kwargs):
    """Draws a plotly chart for this QueryResults.

    Args:
      kind: The kind of chart to draw. Defaults to "linechart".
      partition_by: One or more labels to partition the results into separate
        charts. It can be a string or a list/tuple. Defaults to 'metric_type'.
      annotate_by: One or more labels to aggregate and annotate each chart by.
        It can be a string or a list/tuple.
      aggregation_method: The method to apply to the aggregate the timeseries in
        a single chart given the annotate_by fields.
      **kwargs: Keyword arguments to pass in to the underlying visualization.

    Raises:
      ValueError: "kind" is not a valid plot kind.
    """
    if kind not in self._ALL_PLOT_KINDS:
      raise ValueError('%r is not a valid plot kind' % kind)
    if self.empty:
      return
    if isinstance(aggregation_method, basestring):
      aggregation_method = Aggregator[aggregation_method]

    partition_by = _listify(partition_by)
    annotate_by = _listify(annotate_by)

    if not partition_by:
      dataframe_iter = [(None, self._dataframe)]
    else:
      dataframe_iter = self._dataframe.groupby(level=partition_by, axis=1)

    for name, dataframe in dataframe_iter:
      if not partition_by:
        title = 'All timeseries'
      else:
        annotations = ['%s = %r' % (key, value)
                       for key, value in zip(partition_by, _listify(name))]
        title = ', '.join(annotations)

      if annotate_by is None:
        aggregated_series = getattr(dataframe, aggregation_method)(axis=1)
        dataframe = aggregated_series.to_frame(name='aggregated')
      else:
        grouped = dataframe.groupby(level=annotate_by, axis=1)
        dataframe = grouped.agg(aggregation_method)

      # Call the appropriate visualization function.
      getattr(_visualization, kind)(
          dataframe, labels=annotate_by, title=title, **kwargs)

  def linechart(self, partition_by=_DEFAULT_PARTITION_BY, annotate_by=None,
                aggregation_method='mean', **kwargs):
    """Draws a plotly linechart for this QueryResults.

    Args:
      partition_by: One or more labels to partition the results into separate
        linecharts. It can be a string or a list/tuple. Defaults to 'metric_type'.
      annotate_by: One or more labels to aggregate and annotate each linechart
        by. It can be a string or a list/tuple.
      aggregation_method: The method to apply to the aggregate the timeseries in
        a single chart given the annotate_by fields.
      **kwargs: Any arguments to pass in to the layout engine
        plotly.graph_objs.Layout().
    """
    self.plot('linechart', partition_by, annotate_by, aggregation_method,
              **kwargs)

  def heatmap(self, partition_by=_DEFAULT_PARTITION_BY, annotate_by=None,
              aggregation_method='mean', zrange=None, colorscale=None,
              is_logscale=False, is_divergent=False, **kwargs):
    """Draws a plotly heatmap for this QueryResults.

    Args:
      partition_by: One or more labels to partition the results into separate
        heatmaps. It can be a string or a list/tuple. Defaults to 'metric_type'.
      annotate_by: One or more labels to aggregate and annotate each heatmap
        by. It can be a string or a list/tuple.
      aggregation_method: The method to apply to the aggregate the timeseries in
        a single chart given the annotate_by fields.
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
    self.plot('heatmap', partition_by, annotate_by, aggregation_method,
              zrange=zrange, colorscale=colorscale, is_logscale=is_logscale,
              is_divergent=is_divergent, **kwargs)


def _listify(value):
  """If value is a string, convert to a list of one element."""
  return [value] if isinstance(value, basestring) else value
