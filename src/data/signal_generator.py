import numpy as np
from tqdm import tqdm
from pycbc.waveform import get_td_waveform


class SignalGenerator:
    def __init__(self, config, preprocessor):
        self.config = config
        self.preprocessor = preprocessor

    def generate_signal(self, mass1, mass2):
        """
        mass1: Mass of the first object in solar masses
        mass2: Mass of the second object in solar masses

        :return: Time-domain gravitational wave signal
        """
        hp, _ = get_td_waveform(approximant="IMRPhenomPv2",
                                mass1=mass1,
                                mass2=mass2,
                                delta_t=1.0 / self.config.SAMPLING_RATE,
                                f_lower=20)
        return hp.numpy()

    def generate_signals(self, num_signals, mass1_range, mass2_range):
        """
        num_signals: Number of signals to generate
        mass1_range: Range for mass1 (min, max)
        mass2_range: Range for mass2 (min, max)

        :return: List of generated signals, all padded/truncated to the same length
        """
        signals = []
        print("Generating gravitational wave signals...")
        for _ in tqdm(range(num_signals)):
            mass1 = np.random.uniform(*mass1_range)
            mass2 = np.random.uniform(*mass2_range)
            signal = self.generate_signal(mass1, mass2)
            signal = self.preprocessor.pad_or_truncate(signal)
            signals.append(signal)
        return np.array(signals)
