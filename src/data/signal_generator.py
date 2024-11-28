import torch
from pycbc.waveform import get_td_waveform
from pycbc import waveform


def generate_gw_signal(m1, m2, f_lower=30.0, duration=4, sample_rate=2048):
    """
    Generate a synthetic gravitational wave signal for given component masses.

    Parameters:
    - m1: Mass of the first component in solar masses
    - m2: Mass of the second component in solar masses
    - f_lower: Lower frequency cutoff for the waveform (default 30 Hz)
    - duration: Duration of the signal in seconds (default 4 seconds)
    - sample_rate: Sampling rate for the signal (default 2048 Hz)

    Returns:
    - strain: The synthetic gravitational wave strain as a PyTorch tensor
    """

    hp, hc = get_td_waveform(
        approximant="SEOBNRv4",
        mass1=m1,
        mass2=m2,
        f_lower=f_lower,
        delta_t=1 / sample_rate,
        duration=duration
    )

    strain = hp
    strain_normalized = strain / strain.max()
    strain_tensor = torch.tensor(strain_normalized, dtype=torch.float32)

    return strain_tensor
