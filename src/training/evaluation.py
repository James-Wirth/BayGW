from tqdm import tqdm
import numpy as np

def evaluate_signals(signals, generated_signals):
    overlaps = []
    print("Evaluating signals...")
    for signal, generated_signal in tqdm(zip(signals, generated_signals), total=len(signals)):
        overlaps.append(overlap_integral(signal, generated_signal))
    return overlaps

def overlap_integral(signal1, signal2):
    norm1 = np.linalg.norm(signal1)
    norm2 = np.linalg.norm(signal2)
    return np.dot(signal1, signal2) / (norm1 * norm2)
