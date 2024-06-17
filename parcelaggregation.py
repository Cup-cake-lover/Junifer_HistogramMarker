
from junifer.testing.datagrabbers import (
    OasisVBMTestingDataGrabber    
)
from junifer.datareader import DefaultDataReader
from junifer.markers import ParcelAggregation
from junifer.utils import configure_logging
from typing import Any, Optional, Union, List, Dict
from junifer.api.decorators import register_marker
from junifer.markers.base import BaseMarker
import numpy as np


#@register_marker
class HistogramMarker(BaseMarker):

    _DEPENDENCIES = {"numpy"}

    def __init__(
        self,
        bins: int,
        on: Optional[Union[str, List[str]]] = None,
        name: Optional[str] = None,
    ) -> None:
        self.bins = bins
        super().__init__(on=on, name=name)

    def get_valid_inputs(self) -> List[str]:
        return ["VBM_GM"]

    def get_output_type(self, input_type: str) -> str:
        return "histogram"

    def compute(
        self,
        input: Dict[str, Any],
        extra_input: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Get the data
        data = input["data"]

        # Compute the histogram
        hist, bin_edges = np.histogram(
            data,
            bins=self.bins,

        )
        # Create the output dictionary
        out = {"data": hist, "bin_edges": bin_edges}

        return out


with OasisVBMTestingDataGrabber() as dg:
    # Get the first element
    element = dg.get_elements()[0]
    # Read the element
    element_data = DefaultDataReader().fit_transform(dg[element])
    # Initialize marker
    marker = ParcelAggregation(parcellation="Schaefer400x17", method="mean")
    # Compute feature
    feature = marker.fit_transform(element_data)
    
    

    # Compute histograms and return bins
    # Example implementation.
    histogram = HistogramMarker(bins=100)
    hists = histogram.compute(feature["VBM_GM"])
    
    # The output datafile should be changed!
    np.save("histogram_data.npy",hists)
    print(hists)
    
    