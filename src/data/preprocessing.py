import numpy as np
import torch
from pycbc import waveform
from pycbc.detector import Detector
from pycbc.catalog import find_event_in_catalog


def load_gw_data(event, duration=4, sample_rate=2048):
    """
    Load a gravitational wave event from the PyCBC catalog, then resample and preprocess it.

    Parameters:
    - event: The event name (e.g., 'GW150914')
    - duration: Duration of the signal in seconds (default 4 seconds)
    - sample_rate: Sampling rate for the data (default 2048 Hz)

    Returns:
    - strain: Processed strain data as a PyTorch tensor
    """

    data = find_event_in_catalog(event)
    gps_start = data['GPS start']
    gps_end = gps_start + duration

    strain = data.get('strain', gps_start, gps_end)
    strain_resampled = strain.resample(sample_rate)
    strain_filtered = strain_resampled.highpass_fir(30.0)
    strain_normalized = strain_filtered.data
    strain_tensor = torch.tensor(strain_normalized, dtype=torch.float32)

    return strain_tensor
