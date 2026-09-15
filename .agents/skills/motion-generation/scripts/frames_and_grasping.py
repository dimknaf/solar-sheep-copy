# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frame/offset helpers for tool-frame motion targets."""

import numpy as np


def tool_local_contact_offset(tool_position, tool_rotation_matrix, contact_point):
    """Express a world-space contact point relative to the controller tool frame."""
    return tool_rotation_matrix.T @ (contact_point - tool_position)


def tool_target_from_contact_target(contact_target, tool_rotation_matrix, contact_offset_tool):
    """Convert a desired physical contact point into a controller tool target."""
    return contact_target - tool_rotation_matrix @ contact_offset_tool
