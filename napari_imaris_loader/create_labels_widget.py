from napari_plugin_engine import napari_hook_implementation
import numpy as np
from qtpy.QtWidgets import QWidget, QGridLayout, QPushButton
import zarr
from napari.layers.labels._labels_key_bindings import activate_paint_mode
from napari.layers.labels._labels_utils import indices_in_shape, sphere_indices
from napari.layers.labels import Labels


class AddLabelsLayer(QWidget):
    def __init__(self, napari_viewer):
        self.viewer = napari_viewer
        super().__init__()

        layout = QGridLayout()
        create_btn = QPushButton('Create', self)
        paint_button = QPushButton('Paint', self)

        def trigger_create():
            extent = napari_viewer.layers.extent.world
            scale = napari_viewer.layers.extent.step
            scene_size = extent[1] - extent[0]
            corner = extent[0] + 0.5 * napari_viewer.layers.extent.step
            shape = [
                np.round(s / sc).astype('int') if s > 0 else 1
                for s, sc in zip(scene_size, scale)
            ]
            empty_labels = zarr.zeros(
                shape,
                chunks=napari_viewer.layers[0].data[0].chunksize,  # TODO replace layers[0] by Image layer
                dtype=napari_viewer.layers[0].data[0].dtype
            )
            napari_viewer.add_labels(empty_labels, translate=np.array(corner), scale=scale)

        def trigger_paint():
            Labels.paint = paint
            activate_paint_mode(napari_viewer.layers[-1])  # TODO replace layers[-1] by Labels layer

        create_btn.clicked.connect(trigger_create)
        paint_button.clicked.connect(trigger_paint)
        layout.addWidget(create_btn)
        layout.addWidget(paint_button)

        # activate layout
        self.setLayout(layout)


@napari_hook_implementation(specname='napari_experimental_provide_dock_widget')
def create_labels_layer():
    return AddLabelsLayer


def paint(self, coord, new_label, refresh=True):
    """Paint over existing labels with a new label, using the selected
    brush shape and size, either only on the visible slice or in all
    n dimensions.

    Parameters
    ----------
    coord : sequence of int
        Position of mouse cursor in image coordinates.
    new_label : int
        Value of the new label to be filled in.
    refresh : bool
        Whether to refresh view slice or not. Set to False to batch paint
        calls.
    """
    shape = self.data.shape
    dims_to_paint = sorted(self._dims_order[-self.n_edit_dimensions :])
    dims_not_painted = sorted(self._dims_order[: -self.n_edit_dimensions])
    paint_scale = np.array(
        [self.scale[i] for i in dims_to_paint], dtype=float
    )

    slice_coord = [int(np.round(c)) for c in coord]
    if self.n_edit_dimensions < self.ndim:
        coord_paint = [coord[i] for i in dims_to_paint]
        shape = [shape[i] for i in dims_to_paint]
    else:
        coord_paint = coord

    # Ensure circle doesn't have spurious point
    # on edge by keeping radius as ##.5
    radius = np.floor(self.brush_size / 2) + 0.5
    mask_indices = sphere_indices(radius, tuple(paint_scale))

    mask_indices = mask_indices + np.round(np.array(coord_paint)).astype(
        int
    )

    # discard candidate coordinates that are out of bounds
    mask_indices = indices_in_shape(mask_indices, shape)

    # Transfer valid coordinates to slice_coord,
    # or expand coordinate if 3rd dim in 2D image
    slice_coord_temp = [m for m in mask_indices.T]
    if self.n_edit_dimensions < self.ndim:
        for j, i in enumerate(dims_to_paint):
            slice_coord[i] = slice_coord_temp[j]
        for i in dims_not_painted:
            slice_coord[i] = slice_coord[i] * np.ones(
                mask_indices.shape[0], dtype=int
            )
    else:
        slice_coord = slice_coord_temp

    slice_coord = tuple(slice_coord)

    # Fix indexing for xarray if necessary
    # See http://xarray.pydata.org/en/stable/indexing.html#vectorized-indexing
    # for difference from indexing numpy
    try:
        import xarray as xr

        if isinstance(self.data, xr.DataArray):
            slice_coord = tuple(xr.DataArray(i) for i in slice_coord)
    except ImportError:
        pass

    # slice coord is a tuple of coordinate arrays per dimension
    # subset it if we want to only paint into background/only erase
    # current label
    if self.preserve_labels:
        if new_label == self._background_label:
            keep_coords = self.data[slice_coord] == self.selected_label
        else:
            keep_coords = self.data[slice_coord] == self._background_label
        slice_coord = tuple(sc[keep_coords] for sc in slice_coord)

    if isinstance(self.data, zarr.Array):
        # save the existing values to the history
        self._save_history(
            (
                slice_coord,
                np.array(self.data.vindex[slice_coord], copy=True),
                new_label,
            )
        )
        # update the labels image
        self.data.vindex[slice_coord] = new_label
    else:
        # save the existing values to the history
        self._save_history(
            (
                slice_coord,
                np.array(self.data[slice_coord], copy=True),
                new_label,
            )
        )

        # update the labels image
        self.data[slice_coord] = new_label

    if refresh is True:
        self.refresh()
