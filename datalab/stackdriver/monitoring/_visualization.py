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

"""Visualization methods."""

from __future__ import absolute_import

import collections

import matplotlib.pyplot as plt

import plotly.graph_objs as go
import plotly.offline as py

# Internal imports
from . import _dataframe as dataframe_utils


def _get_colorscale(mpl_colorscale=None, is_divergent=False, is_logscale=False,
                    val_range=None):
  """Returns a colorscale usable by the plotly heatmap method.

  Args:
    mpl_colorscale: A colorscale supported by matplotlib.
    is_divergent: boolean, specifies if the series is divergent.
    is_logscale: boolean. If True, then return a logarithmic colorscale.
    val_range: An optional value range.

  Returns:
    A colorscale compatible with plotly heatmap. We return a scale with 11
    colors to map the specified colorscale. The output is a 2-D array where
    each element is a list containing:
      a. A float between 0 and 1
      b. The rgb color as a string of the format: 'rgb(<red>, <green>, <blue)'
    An example based on the colorscale 'RdBu':
    [
      [0.0, 'rgb(103,0,31)'],
      [0.1, 'rgb(176,23,42)'],
      [0.2, 'rgb(214,96,77)'],
      [0.3, 'rgb(243,163,128)'],
      [0.4, 'rgb(253,219,199)'],
      [0.5, 'rgb(246,246,246)'],
      [0.6, 'rgb(209,229,240)'],
      [0.7, 'rgb(144,196,221)'],
      [0.8, 'rgb(67,147,195)'],
      [0.9, 'rgb(32,100,170)'],
      [1.0, 'rgb(5,48,97)']
   ]
  """
  if mpl_colorscale is None:
    if not is_divergent and val_range is not None:
      # If the colorscale is not marked as divergent, then decide based on the
      # range.
      df_min, df_max = val_range  # pylint: disable=unpacking-non-sequence

      # Maximum ratio allowed that marks a series as divergent.
      max_ratio = 5
      is_divergent = (df_min < 0 and df_max > 0 and
                      1.0/max_ratio <= abs(df_min)/float(df_max) <= max_ratio)

    if is_divergent:
      cmap = plt.get_cmap('RdBu')
    else:
      cmap = plt.get_cmap('GnBu')
  else:
    cmap = plt.get_cmap(mpl_colorscale)

  num_colors = 11
  num_partitions = num_colors - 1
  int_rgb = cmap([float(i)/num_partitions for i in xrange(num_colors)],
                 bytes=True)[:, :3]

  rgb_colormap = []
  for i, row in enumerate(int_rgb.astype(str)):
    if is_logscale:
      value = 10 ** -(num_partitions - i) if i > 0 else 0
    else:
      value = float(i)/num_partitions
    rgb_colormap.append([value, 'rgb(%s)' % ','.join(row)])

  return rgb_colormap


def heatmap(df, label=None, labels=None, zrange=None, mpl_colorscale=None,
            is_logscale=False, is_divergent=False, **kwargs):
  """Draws a plotly heatmap for the specified dataframe.

  Args:
    df: The pandas DataFrame object to draw as a heatmap.
    label: str, A single level of column header to pick.
    labels: lsit, one or more levels of column header to pick.
    zrange: A list or tuple of length 2 containing the range to use for the
      colormap. If not specified, then it is calculated from the dataframe.
    mpl_colorscale: str, A colorscale supported by matplotlib. See:
      http://matplotlib.org/examples/color/colormaps_reference.html
    is_logscale: boolean, if True, then a logarithmic colorscale is used.
    is_divergent: boolean, specifies if the data has diverging values. If not
      manually set to True, we check if the data diverges around 0, and use an
      appropriate default colormap. Ignored if you specify the colormap.
    **kwargs: Any arguments to pass in to the layout engine
      plotly.graph_objs.Layout().
  """
  if df is None or df.empty:
    return
  # For the correlation matrix, the index and columns are the same. So we make
  # the index be a single level as well if required.
  if len(df.index.names) > 1 and df.index.names == df.columns.names:
    df = dataframe_utils.extract_single_level(df.T, label, labels).T

  df = dataframe_utils.extract_single_level(df, label, labels)

  # Make the heatmap taller.
  if 'height' not in kwargs:
    kwargs['height'] = 800

  # Make the left margin bigger
  if 'margin' not in kwargs:
    kwargs['margin'] = {}
  if 'l' not in kwargs['margin']:
    kwargs['margin']['l'] = 150

  zrange = zrange or (df.min().min(), df.max().max())
  colorscale = _get_colorscale(
      mpl_colorscale, is_divergent, is_logscale, zrange)
  heatmap_data = go.Heatmap(
      z=df.T.values.tolist(), x=df.index, y=df.columns, colorscale=colorscale,
      zauto=False, zmin=zrange[0], zmax=zrange[1])

  fig = go.Figure(data=[heatmap_data], layout=go.Layout(**kwargs))
  py.iplot(fig, show_link=False)


def linechart(df, label=None, labels=None, **kwargs):
  """Draws a plotly linechart for the specified dataframe.

  Args:
    df: The pandas DataFrame object to draw as a linechart.
    label: str, A single level of column header to pick.
    labels: lsit, one or more levels of column header to pick.
    **kwargs: Any arguments to pass in to the layout engine
      plotly.graph_objs.Layout().
  """
  if df is None or df.empty:
    return
  df = dataframe_utils.extract_single_level(df, label, labels)
  column_names = df.columns.tolist()
  if len(column_names) > len(set(column_names)):
    duplicates = [(col, count) for col, count in
                  collections.Counter(column_names).iteritems() if count > 1]
    raise ValueError('Cannot draw linechart of a dataframe with duplicate '
                     'column headers: %s' % duplicates)

  if 'height' not in kwargs:
    kwargs['height'] = 600

  data = [go.Scatter(x=df.index, y=df[col], name=col) for col in df.columns]
  fig = go.Figure(data=data, layout=go.Layout(**kwargs))
  py.iplot(fig, show_link=False)
