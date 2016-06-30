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

"""pandas dataframe utility methods."""

from __future__ import absolute_import
from past.builtins import basestring

import pandas

from . import _utils


def _dedup_columns(columns):
  """Deduplicates columns by adding a suffix to duplicate columns."""
  num_cols = len(columns)
  deduped_columns = [None] * num_cols
  count = 0
  matched = None
  for i, col in enumerate(columns):
    if matched == col:
      count += 1
      col += '#%d' % count
    else:
      matched = col
      count = 1
      if i < (num_cols - 1) and col == columns[i + 1]:
        col += '#%d' % count
    deduped_columns[i] = col
  return deduped_columns


def extract_single_level(dataframe, levels=None):
  """Returns a new dataframe with a single level of column header.

  Args:
    dataframe: The pandas DataFrame object that we do the manipulation on.
    levels: A list of one or more levels of column header to pick.

  Returns:
    A new pandas dataframe with the same data as the input dataframe, but with
    a single level for the column header. If a single level is specified, then
    that level becomes the new single level column header. Otherwise, two or
    more levels are combined using ', ' as a separator in the order specified.

    Note: Columns are reordered to have the headers in an alphabetical order.
  """
  df_single = extract_levels(dataframe, levels)

  if len(df_single.columns.names) > 1:
    df_single.columns = [', '.join(map(str, col))
                         for col in df_single.columns.values]
  df_single.columns = _dedup_columns(df_single.columns)
  return df_single


def extract_levels(dataframe, levels=None):
  """Returns a new dataframe with the column headers of interest for a user.

  Args:
    dataframe: The pandas DataFrame object that we do the manipulation on.
    levels: A list of one or more levels of column header to pick.

  Returns:
    A new pandas dataframe with the same data as the input dataframe, but with
    only the levels that the user specifies. If one or more levels specified
    by the user do not exist in the input dataframe, then they will be inserted,
    and the values will be the empty string.

    Note: Columns are reordered to have the headers in an alphabetical order.
  """
  levels = _utils.listify(levels)
  df_mult = dataframe.copy()

  # Checking for cases where nothing needs to be done: either the levels are
  # empty, or the levels match the columns.
  if not levels or (levels == dataframe.columns.names):
    return df_mult

  if len(levels) == 1:
    # Extracting a single level can be done in one line.
    df_mult.columns = dataframe.columns.get_level_values(levels[0])
  else:
    # Extracting multiple levels is more complex - we convert the headers
    # into a dataframe first, and then extract the levels in order.
    df_headers = pandas.DataFrame(
        dataframe.columns.tolist(), columns=dataframe.columns.names)
    df_headers = df_headers.reindex(columns=levels).fillna('')
    df_mult.columns = pandas.MultiIndex.from_arrays(
        df_headers.T.values, names=df_headers.columns.tolist())

  return df_mult.sort_index(axis=1)


def add_level(dataframe, level, level_name=None):
  if isinstance(level, basestring):
    level = [level] * len(dataframe.columns)
  df_headers = pandas.DataFrame(dataframe.columns.tolist(),
                                columns=dataframe.columns.names)
  df_headers.insert(0, level_name, level)
  new_columns =  pandas.MultiIndex.from_arrays(
      df_headers.T.values, names=df_headers.columns.tolist())
  return pandas.DataFrame(dataframe.values, index=dataframe.index,
                          columns=new_columns)
