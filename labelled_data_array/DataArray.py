import numpy as np
import pandas as pd

from typing import Tuple, Sequence


class DataArray:

    def __init__(self, values: np.ndarray, axis_labels: Tuple[Sequence[str]]):
        self.values = values
        self.axis_labels = [np.asarray(cur_labels) for cur_labels in axis_labels]

        # TODO: Check that the axis labels are unique

        if len(self.axis_labels) != self.values.ndim:
            raise ValueError(
                f"Expected {self.values.ndim} axis labels, got {len(self.axis_labels)}"
            )
        self._n_dims = len(self.axis_labels)

        for ix in range(self._n_dims):
            if self.axis_labels[ix].size != values.shape[ix]:
                raise ValueError(
                    f"Expected {self.values.shape[ix]} axis labels for axis {ix}, "
                    f"got {self.axis_labels[ix].size}"
                )

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
        def __init__(self, parent: "DataArray"):
            self.parent = parent

        def __getitem__(self, key):
            if not isinstance(key, tuple):
                key = (key,)

            if len(key) != self.parent._n_dims:
                raise ValueError(
                    f"Expected {self.parent._n_dims} indices, got {len(key)}"
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
