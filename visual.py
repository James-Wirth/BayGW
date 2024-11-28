import torch
import matplotlib.pyplot as plt
from src.models.normalizing_flow import RealNVP


def load_model(model_path, input_dim, hidden_dim, conditioning_dim, device="cpu"):
    """
    Load the trained RealNVP model.

    Parameters:
    - model_path: Path to the saved model
    - input_dim: Input dimension of the model
    - hidden_dim: Hidden dimension of the model
    - conditioning_dim: Conditioning dimension (e.g., masses)
    - device: Device to load the model on (default: "cpu")

    Returns:
    - model: Loaded RealNVP model
    """
    model = RealNVP(input_dim=input_dim, hidden_dim=hidden_dim, conditioning_dim=conditioning_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def generate_and_visualize_samples(model, num_samples, conditioning_params, device="cpu"):
    """
    Generate and visualize new samples from the model.

    Parameters:
    - model: Trained RealNVP model
    - num_samples: Number of samples to generate
    - conditioning_params: Conditioning parameters for sample generation (e.g., masses)
    - device: Device to run the generation on
    """
    conditioning_params = torch.tensor(conditioning_params, dtype=torch.float32, device=device)

    samples = model.sample(num_samples, conditioning=conditioning_params)

    plt.figure(figsize=(12, 6))
    for i, sample in enumerate(samples):
        plt.plot(sample.detach().cpu().numpy(), label=f"Sample {i + 1}")
    plt.title("Generated Gravitational Wave Signals")
    plt.xlabel("Time Steps")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()


def test_generate_samples():
    """
    Test function to load the model, generate new signals, and visualize them.
    """
    model_path = "realnvp_model.pth"
    input_dim = 4096
    hidden_dim = 512
    conditioning_dim = 2
    device = "cpu"

    print("Loading the trained model...")
    model = load_model(model_path, input_dim, hidden_dim, conditioning_dim, device)
    print("Model loaded")

    num_samples = 1
    conditioning_params = [[30.0, 25.0] for _ in range(num_samples)]  # Repeat conditioning for all samples

    print("Generating and visualizing new samples...")
    generate_and_visualize_samples(model, num_samples, conditioning_params, device)


if __name__ == "__main__":
    test_generate_samples()
