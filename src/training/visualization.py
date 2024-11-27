import matplotlib.pyplot as plt

def plot_signal(signal, title="Signal", save_path=None):
    plt.figure(figsize=(10, 4))
    plt.plot(signal)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def compare_signals(signal1, signal2, labels=("Signal 1", "Signal 2"), save_path=None):
    plt.figure(figsize=(10, 4))
    plt.plot(signal1, label=labels[0])
    plt.plot(signal2, label=labels[1], linestyle='dashed')
    plt.title("Signal Comparison")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
