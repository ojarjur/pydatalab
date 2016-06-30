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

"""Provides utility methods for the Monitoring API."""

from __future__ import absolute_import

import collections

import gcloud.monitoring

import datalab.context


def make_client(project_id=None, context=None):
  context = context or datalab.context.Context.default()
  project_id = project_id or context.project_id
  return gcloud.monitoring.Client(
      project=project_id,
      credentials=context.credentials,
  )


def list_resource_descriptors(
    filter_string=None, project_id=None, context=None):
  """List all monitored resource descriptors for the project.

  Args:
    filter_string: An optional filter expression describing the resource
      descriptors to be returned.
    project_id: An optional project ID or number to override the one provided
        by the context.
    context: An optional Context object to use instead of the global default.

  Returns:
    A list of resource descriptor instances.
  """
  client = make_client(project_id, context)
  return client.list_resource_descriptors(filter_string)


def list_metric_descriptors(
    filter_string=None, type_prefix=None, project_id=None, context=None):
  """List all metric descriptors for the project.

  Args:
    filter_string: An optional filter expression describing the resource
      descriptors to be returned.
    type_prefix: An optional prefix constraining the selected metric types. This
      adds ``metric.type = starts_with("<prefix>")`` to the filter.
    project_id: An optional project ID or number to override the one provided
        by the context.
    context: An optional Context object to use instead of the global default.

  Returns:
    A list of metric descriptor instances.
  """
  client = make_client(project_id, context)
  return client.list_metric_descriptors(filter_string, type_prefix)


def listify(value):
  """If value is a string, convert to a list of one element."""
  if value is None:
    return []
  elif isinstance(value, basestring):
    return [value]
  elif isinstance(value, collections.Iterable):
    return value
  else:
    raise TypeError('"value" must be a string or an iterable')
