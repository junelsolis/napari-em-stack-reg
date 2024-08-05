import os

import dask.array as da
import numpy as np
from napari.layers import Image
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari


class StackableImage:
    def __init__(self, viewer: "napari.viewer.Viewer"):
        self._viewer = viewer
        self._set_original_stack()

        self._reference_index = 0
        self._moving_index = 1
        self._transforms = []

    def _set_original_stack(self):
        image_layer = next(
            (
                layer
                for layer in self._viewer.layers
                if isinstance(layer, Image)
            ),
            None,
        )

        if image_layer is not None:
            if isinstance(image_layer.data, np.ndarray):
                self._original_stack = da.from_array(image_layer.data)
            elif isinstance(image_layer.data, da.Array):
                self._original_stack = image_layer.data

    def get_registration_images(self, reference_index: int = 0):
        self._reference_img = self._original_stack[reference_index]
        self._moving_img = self._original_stack[self._moving_index]

        current_image_layer = next(
            (
                layer
                for layer in self._viewer.layers
                if isinstance(layer, Image)
            ),
            None,
        )

        if current_image_layer is not None:
            self._viewer.layers.remove(current_image_layer)

        # add reference image
        self._viewer.add_image(
            self._reference_img,
            name=f"ref - slice {reference_index}",
            blending="translucent_no_depth",
            colormap="gray",
        )

        # add moving image
        self._viewer.add_image(
            self._moving_img,
            name=f"moving - slice {self._moving_index}",
            blending="translucent_no_depth",
            colormap="gray",
            opacity=0.5,
        )

        self._viewer.layers[-1].mode = "transform"
