"""Provide class for HistogramMarker."""

# Authors: Hari Prasad SV
# License: AGPL

from typing import Any, ClassVar, Dict, List, Optional, Union

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker
from junifer.utils import logger
from junifer.data import get_mask
from nilearn.maskers import NiftiMasker

__all__ = ["HistogramMarker"]

@register_marker
class HistogramMarker(BaseMarker):
    """Class for histogram marker.

    Parameters
    ----------
    bins : positive int, optional
        The number of equal-width bins in the given range (default 10).
    name : str, optional
        The name of the marker. If None, will use ``VBM_GM_HistogramMarker``
        (default None).
    masks : str, dict, list of (dict or str), or None, optional
        The masks to be used for computation (default None).

    """

    _DEPENDENCIES = {"nilearn","numpy"}

    _MARKER_INOUT_MAPPINGS: ClassVar[Dict[str, Dict[str, str]]] = {
        "VBM_GM": {
            "hist": "vector",
            "bin_edges": "vector",
        },
    }

    def __init__(
        self,
        bins: int,
        name: Optional[str] = None,
        masks: Union[str, Dict, List[Union[Dict, str]], None] = None
    ) -> None:
        self.bins = bins
        self.masks = masks
        super().__init__(on="VBM_GM", name=name)

    def compute(
        self,
        input: Dict[str, Any],
        extra_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute.

        Parameters
        ----------
        input : dict
            The VBM_GM data as dictionary.
        extra_input : dict, optional
            The other fields in the pipeline data object (default None).

        Returns
        -------
        dict
            The computed result as dictionary. The dictionary has the following
            keys:

            * ``hist`` : dict of histogram bin count (int) and histogram values
                         (``np.ndarray``)
            * ``bin_edges`` : dict of histogram bin count (int) and bin edges
                              (``np.ndarray``)

        """
        logger.debug("Computing histogram")

        t_input_img = input["data"]
        # Load mask if provided
        if self.masks is not None:
            logger.debug(f"Masking with {self.masks}")
            breakpoint()
            # Get tailored mask
            mask_img = get_mask(
                masks=self.masks, target_data=input, extra_input=extra_input
            )
            
            # Apply the mask to the input image
            logger.debug("Masking")
            masker = NiftiMasker(mask_img, target_affine=t_input_img.affine)
            data = masker.fit_transform(t_input_img)
            # Flatten the masked data for histogram computation
            data = data.ravel()
        else:
            data = t_input_img.get_fdata().ravel()
        logger.debug("computed masks")    
        
        # Compute the histogram
        hist, bin_edges = np.histogram(
            data,
            bins=self.bins,
        )

        # Create the output dictionary
        return {
            "hist": {
                "data": hist,
                "col_names": list(range(hist.size)),
            },
            "bin_edges": {
                "data": bin_edges,
                "col_names": list(range(bin_edges.size)),
            },
        }
