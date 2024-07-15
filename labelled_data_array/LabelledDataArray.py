import numpy as np
import pandas as pd

from typing import Sequence


class LabelledDataArray:

    def __init__(self, values: np.ndarray, axis_labels: tuple, axis_names: tuple = None):
        self.values = values
        self.axis_labels = [np.asarray(cur_labels) for cur_labels in axis_labels]
        self.axis_names = [f"axis_{ix}" for ix in range(len(axis_labels))] if axis_names is None else list(axis_names)

        # TODO: Check that the axis labels are unique
        # TODO: Add support for setting items
        # Todo: Add support for creating empty DataArray

        if len(self.axis_labels) != self.values.ndim:
            raise ValueError(
                f"Expected {self.values.ndim} axis labels, got {len(self.axis_labels)}"
            )
        self.n_dims = len(self.axis_labels)

        for ix in range(self.n_dims):
            if self.axis_labels[ix].size != values.shape[ix]:
                raise ValueError(
                    f"Expected {self.values.shape[ix]} axis labels for axis {ix}, "
                    f"got {self.axis_labels[ix].size}"
                )

        if not isinstance(self.axis_names, list):
            raise ValueError("axis-names must be a list of strings")

        # TODO: Check that the axis labels are strings

    def __array__(self):
        return self.values

    def __repr__(self):
        return f"DataArray({self.values}, {self.axis_labels})"

    def __getitem__(self, key):
        return self.values[key]

    def __setitem__(self, key, value):
        self.values[key] = value

    class _SelectIndexer:
        def __init__(self, parent: "LabelledDataArray"):
            self.parent = parent

        def __getitem__(self, key):
            if not isinstance(key, tuple):
                key = (key,)

            if len(key) != self.parent.n_dims:
                raise ValueError(
                    f"Expected {self.parent.n_dims} indices, got {len(key)}"
                )

            apply_keys = []
            sel_axis, slice_axis = [], []
            for axis_ix, cur_key in enumerate(key):
                if isinstance(cur_key, slice):
                    apply_keys.append(cur_key)
                    slice_axis.append(axis_ix)

                if isinstance(cur_key, str):
                    if cur_key not in self.parent.axis_labels[axis_ix]:
                        raise ValueError(f"Key '{cur_key}' not found in axis {axis_ix}")

                    apply_keys.append(
                        np.flatnonzero(self.parent.axis_labels[axis_ix] == cur_key)[0]
                    )
                    sel_axis.append(axis_ix)

            apply_keys = tuple(apply_keys)
            result = self.parent[apply_keys]

            if len(result.shape) == 1:
                assert len(slice_axis) == 1

                return pd.Series(
                    data=result,
                    index=self.parent.axis_labels[slice_axis[0]][
                        apply_keys[slice_axis[0]]
                    ],
                )
            if len(result.shape) == 2:
                assert len(slice_axis) == 2

                return pd.DataFrame(
                    data=result,
                    index=self.parent.axis_labels[slice_axis[0]][
                        apply_keys[slice_axis[0]]
                    ],
                    columns=self.parent.axis_labels[slice_axis[1]][
                        apply_keys[slice_axis[1]]
                    ],
                )
            else:
                # Todo: Return another DataArray
                return result

    @property
    def sel(self):
        return self._SelectIndexer(self)

    class _LabelIndexer:
        def __init__(self, parent: "LabelledDataArray"):
            self.parent = parent

        def __getitem__(self, key):
            if not isinstance(key, str):
                raise ValueError("Expected a string key")

            if key not in self.parent.axis_names:
                raise ValueError(f"Key '{key}' not found in axis names {self.parent.axis_names}")

            return self.parent.axis_labels[self.parent.axis_names.index(key)]

    @property
    def labels(self):
        """Returns the labels for the specified axis"""
        return self._LabelIndexer(self)

    @property
    def shape(self):
        return self.values.shape
