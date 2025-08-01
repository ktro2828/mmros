# Copyright 2025 Kotaro Uetake.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import numpy as np


class ColorMap:
    """ColorMap class for managing color palettes."""

    COLORS = (
        np.array(
            [
                (0.000, 0.447, 0.741),
                (0.850, 0.325, 0.098),
                (0.929, 0.694, 0.125),
                (0.494, 0.184, 0.556),
                (0.466, 0.674, 0.188),
                (0.301, 0.745, 0.933),
                (0.635, 0.078, 0.184),
                (0.300, 0.300, 0.300),
                (0.600, 0.600, 0.600),
                (1.000, 0.000, 0.000),
                (1.000, 0.500, 0.000),
                (0.749, 0.749, 0.000),
                (0.000, 1.000, 0.000),
                (0.000, 0.000, 1.000),
                (0.667, 0.000, 1.000),
                (0.333, 0.333, 0.000),
                (0.333, 0.667, 0.000),
                (0.333, 1.000, 0.000),
                (0.667, 0.333, 0.000),
                (0.667, 0.667, 0.000),
                (0.667, 1.000, 0.000),
                (1.000, 0.333, 0.000),
                (1.000, 0.667, 0.000),
                (1.000, 1.000, 0.000),
                (0.000, 0.333, 0.500),
                (0.000, 0.667, 0.500),
                (0.000, 1.000, 0.500),
                (0.333, 0.000, 0.500),
                (0.333, 0.333, 0.500),
                (0.333, 0.667, 0.500),
                (0.333, 1.000, 0.500),
                (0.667, 0.000, 0.500),
                (0.667, 0.333, 0.500),
                (0.667, 0.667, 0.500),
                (0.667, 1.000, 0.500),
                (1.000, 0.000, 0.500),
                (1.000, 0.333, 0.500),
                (1.000, 0.667, 0.500),
                (1.000, 1.000, 0.500),
                (0.000, 0.333, 1.000),
                (0.000, 0.667, 1.000),
                (0.000, 1.000, 1.000),
                (0.333, 0.000, 1.000),
                (0.333, 0.333, 1.000),
                (0.333, 0.667, 1.000),
                (0.333, 1.000, 1.000),
                (0.667, 0.000, 1.000),
                (0.667, 0.333, 1.000),
                (0.667, 0.667, 1.000),
                (0.667, 1.000, 1.000),
                (1.000, 0.000, 1.000),
                (1.000, 0.333, 1.000),
                (1.000, 0.667, 1.000),
                (0.333, 0.000, 0.000),
                (0.500, 0.000, 0.000),
                (0.667, 0.000, 0.000),
                (0.833, 0.000, 0.000),
                (1.000, 0.000, 0.000),
                (0.000, 0.167, 0.000),
                (0.000, 0.333, 0.000),
                (0.000, 0.500, 0.000),
                (0.000, 0.667, 0.000),
                (0.000, 0.833, 0.000),
                (0.000, 1.000, 0.000),
                (0.000, 0.000, 0.167),
                (0.000, 0.000, 0.333),
                (0.000, 0.000, 0.500),
                (0.000, 0.000, 0.667),
                (0.000, 0.000, 0.833),
                (0.000, 0.000, 1.000),
                (0.000, 0.000, 0.000),
                (0.143, 0.143, 0.143),
                (0.286, 0.286, 0.286),
                (0.429, 0.429, 0.429),
                (0.571, 0.571, 0.571),
                (0.714, 0.714, 0.714),
                (0.857, 0.857, 0.857),
                (0.000, 0.447, 0.741),
                (0.314, 0.717, 0.741),
                (0.50, 0.5, 0),
            ],
        )
        .astype(np.float32)
        .reshape(-1, 3)
    )

    @classmethod
    def get_random_rgb(
        cls, *, normalize: bool = False
    ) -> tuple[int, int, int] | tuple[float, float, float]:
        """Return color in RGB randomly.

        Args:
        ----
            normalize (bool, optional): Whether to return normalized color. Defaults to False.

        Returns:
        -------
            tuple[int, int, int] | tuple[float, float, float]: RGB color.

        """
        rgb = np.random.rand(3)
        if normalize:
            return tuple(rgb.tolist())
        else:
            return tuple((rgb * 244).astype(np.uint8).tolist())

    @classmethod
    def get_rgb(
        cls, index: int, *, normalize: bool = False
    ) -> tuple[int, int, int] | tuple[float, float, float]:
        """Return color in RGB chosen from `cls.COLORS`.

        Args:
        ----
            index (int): An index of COLORS. There are 80 colors defined as class variable.
                If the input index is over 80, the number of mod80 is used.
            normalize (bool, optional): Whether to return normalized color. Defaults to False.

        Returns:
        -------
            tuple[int, int, int] | tuple[float, float, float]: RGB color.

        """
        index: int = index % 80
        if normalize:
            return tuple(cls.COLORS[index].tolist())
        else:
            return tuple((cls.COLORS[index] * 255).astype(np.uint8).tolist())
