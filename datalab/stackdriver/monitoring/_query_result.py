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

"""QueryResult object with visualization methods."""

from __future__ import absolute_import

from . import _visualization

class QueryResult(object):
  """QueryResult object contains the results of executing a query."""
  _DEFAULT_GROUP_BY = ('metric_type',)
  _ALL_PLOT_KINDS = ('linechart', 'heatmap')

  def __init__(self, query):
    self._dataframe = query.as_dataframe()
    self._metric_types = query.metric_type

  @property
  def metric_types(self):
    return self._metric_types

  def plot(self, kind='linechart', group_by=_DEFAULT_GROUP_BY,
           annotate_by=None, **kwargs):
    """Draws a plotly chart for the specified dataframe.

    Args:
      kind: The kind of chart to draw. Defaults to "linechart".
      group_by: A list of labels to group the timeseries by.
      annotate_by: A list of labels to annotate each chart by.
      **kwargs: Any arguments to pass in to the layout engine
        plotly.graph_objs.Layout().
    """
    if kind not in self._ALL_PLOT_KINDS:
      raise ValueError('%r is not a valid plot kind' % kind)

    for name, df in self._dataframe.groupby(level=group_by, axis=1):
      if df is None or df.empty:
        continue
      annotations = ['%s = %r' % (key, value)
                     for key, value in zip(annotate_by, name)]
      title = ', '.join(annotations)
      if kind == 'linechart':
        _visualization.linechart(df, labels=annotate_by, title=title)
      elif kind == 'heatmap':
        _visualization.heatmap(df, labels=annotate_by, title=title, **kwargs)

  def linechart(self, group_by=_DEFAULT_GROUP_BY, annotate_by=None, **kwargs):
    """Draws a plotly linechart for the specified dataframe.

    Args:
      group_by: A list of labels to group the timeseries by.
      annotate_by: A list of labels to annotate each linechart by.
      **kwargs: Any arguments to pass in to the layout engine
        plotly.graph_objs.Layout().
    """
    self.plot('linechart', group_by, annotate_by, **kwargs)

  def heatmap(self, group_by=_DEFAULT_GROUP_BY, annotate_by=None, **kwargs):
    """Draws a plotly heatmap for the specified dataframe.

    Args:
      group_by: A list of labels to group the timeseries by.
      annotate_by: A list of labels to annotate each linechart by.
      **kwargs: Any arguments to pass in to the layout engine
        plotly.graph_objs.Layout().
    """
    self.plot('heatmap', group_by, annotate_by, **kwargs)
