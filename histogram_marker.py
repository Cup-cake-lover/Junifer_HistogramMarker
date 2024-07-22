"""Provide class for HistogramMarker."""

# Authors: Hari Prasad SV
# License: AGPL

from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, ClassVar, Set, List, Dict, Union

import numpy as np
from junifer.api.decorators import register_marker
from junifer.markers import BaseMarker
from junifer.utils import logger, raise_error, warn_with_log
from junifer.data import get_mask
from nilearn.image import math_img
from nilearn.maskers import NiftiMasker

if TYPE_CHECKING:
    from junifer.storage import BaseFeatureStorage

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

    """

    _DEPENDENCIES: ClassVar[Set[str]] = {"numpy"}

    def __init__(
        self,
        bins: int,
        name: Optional[str] = None,
        masks: Union[str, Dict, List[Union[Dict, str]], None] = None
    ) -> None:
        self.bins = bins
        self.masks = masks
        super().__init__(on="VBM_GM", name=name)

    def get_valid_inputs(self) -> List[str]:
        """Get valid data types for input.

        Returns
        -------
        list of str
            The list of data types that can be used as input for this marker.

        """
        return ["VBM_GM"]

    def get_output_type(self, input_type: str) -> str:
        """Get output type.

        Parameters
        ----------
        input_type : str
            The data type input to the marker.

        Returns
        -------
        str
            The storage type output by the marker.

        """
        return "vector"

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
        
        # Load mask+
        
        if self.masks is not None:
            logger.debug(f"Masking with {self.masks}")
            # Get tailored mask
            breakpoint()
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

    def store(
        self,
        type_: str,
        feature: str,
        out: Dict[str, Any],
        storage: "BaseFeatureStorage",
    ) -> None:
        """Store.

        Parameters
        ----------
        type_ : str
            The data type to store.
        feature : {"hist", "bin_edges"}
            The feature name to store.
        out : dict
            The computed result as a dictionary to store.
        storage : storage-like
            The storage class, for example, SQLiteFeatureStorage.

        Raises
        ------
        ValueError
            If ``feature`` is invalid.

        """
        if feature in ["hist", "bin_edges"]:
            output_type = "vector"
        else:
            raise_error(f"Unknown feature: {feature}")

        logger.debug(f"Storing {output_type} in {storage}")
        storage.store(kind=output_type, **out)

    def _fit_transform(
        self,
        input: Dict[str, Dict],
        storage: Optional["BaseFeatureStorage"] = None,
    ) -> Dict:
        """Fit and transform.

        Parameters
        ----------
        input : dict
            The Junifer Data object.
        storage : storage-like, optional
            The storage class, for example, SQLiteFeatureStorage.

        Returns
        -------
        dict
            The processed output as a dictionary. If `storage` is provided,
            empty dictionary is returned.

        """
        out = {}
        for type_ in self._on:
            if type_ in input.keys():
                logger.info(f"Computing {type_}")
                t_input = input[type_]
                extra_input = input.copy()
                extra_input.pop(type_)
                t_meta = t_input["meta"].copy()
                t_meta["type"] = type_

                # Returns multiple features
                t_out = self.compute(input=t_input, extra_input=extra_input)

                if storage is None:
                    out[type_] = {}

                for feature_name, feature_data in t_out.items():
                    # Make deep copy of the feature data for manipulation
                    feature_data_copy = deepcopy(feature_data)
                    # Make deep copy of metadata and add to feature data
                    feature_data_copy["meta"] = deepcopy(t_meta)
                    # Update metadata for the feature,
                    # feature data is not manipulated, only meta
                    self.update_meta(feature_data_copy, "marker")
                    # Update marker feature's metadata name
                    feature_data_copy["meta"]["marker"]["name"] += f"_{feature_name}"

                    if storage is not None:
                        logger.info(f"Storing in {storage}")
                        self.store(
                            type_=type_,
                            feature=feature_name,
                            out=feature_data_copy,
                            storage=storage,
                        )
                    else:
                        logger.info("No storage specified, returning dictionary")
                        out[type_][feature_name] = feature_data_copy

        return out
