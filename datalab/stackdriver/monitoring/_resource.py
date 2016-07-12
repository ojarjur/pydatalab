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

"""Provides the ResourceDescriptors in the monitoring API."""

from __future__ import absolute_import

import collections
import fnmatch

from . import _utils

class ResourceDescriptors(object):
  """ResourceDescriptors object for retrieving the resource descriptors."""

  def __init__(self, filter_string=None,
               project_id=None, context=None):
    """Initializes the ResourceDescriptors based on the specified filters.

    Args:
      filter_string: An optional filter expression describing the resource
        descriptors to be returned.
      project_id: An optional project ID or number to override the one provided
          by the context.
      context: An optional Context object to use instead of the global default.

    Returns:
      A list of resource descriptor instances.
    """
    self._client = _utils.make_client(project_id, context)
    self._filter_string = filter_string
    self._descriptors = None

  def list(self, pattern='*'):
    """Returns a list of resource descriptors that match the filters.

    Args:
      pattern: An optional pattern to further filter the descriptors. This can
        include Unix shell-style wildcards. E.g. "aws*", "*cluster*".
    """
    if self._descriptors is None:
      self._descriptors = self._client.list_resource_descriptors(
          self._filter_string)
    return [resource for resource in self._descriptors
            if fnmatch.fnmatch(resource.type, pattern)]

  def table(self, pattern='*', max_rows=-1):
    """Visualize the matching descriptors as an HTML table.

    Args:
      pattern: An optional pattern to further filter the descriptors. This can
        include Unix shell-style wildcards. E.g. "aws*", "*cluster*".
      max_rows: The maximum number of rows (timestamps) to display. A value less
        than 0 shows all rows.
    """
    import IPython.core.display
    import datalab.utils.commands

    data = []
    for i, resource in enumerate(self.list(pattern)):
      if max_rows >= 0 and i >= max_rows:
        break
      data.append(
          collections.OrderedDict([
              ('Resource type', resource.type),
              ('Labels', ', '. join([l.key for l in resource.labels])),
          ])
      )
    return IPython.core.display.HTML(
        datalab.utils.commands.HtmlBuilder.render_table(data))
