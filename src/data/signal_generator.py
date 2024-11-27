from tqdm import tqdm
from pycbc.waveform import get_td_waveform
from pycbc.noise import noise_from_psd
from pycbc.psd.analytical import aLIGOZeroDetHighPower
import numpy as np

class SignalGenerator:
    def __init__(self, config):
        self.config = config

    def generate_signal(self, mass1, mass2):
        hp, _ = get_td_waveform(approximant="IMRPhenomPv2",
                                mass1=mass1,
                                mass2=mass2,
                                delta_t=1.0 / self.config.SAMPLING_RATE,
                                f_lower=20)
        return hp.numpy()

    def add_noise(self, signal):
        psd = aLIGOZeroDetHighPower(len(signal), 1.0 / self.config.SAMPLING_RATE, f_low=20)
        noise = noise_from_psd(len(signal), delta_t=1.0 / self.config.SAMPLING_RATE, psd=psd)
        return signal + noise.numpy()

    def generate_signals_with_noise(self, num_signals, mass1_range, mass2_range):
        signals = []
        print("Generating signals with noise...")
        for _ in tqdm(range(num_signals)):
            mass1 = np.random.uniform(*mass1_range)
            mass2 = np.random.uniform(*mass2_range)
            signal = self.generate_signal(mass1, mass2)
            signal_with_noise = self.add_noise(signal)
            signals.append(signal_with_noise)
        return signals
