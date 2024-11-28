import argparse
import torch
from torch.utils.data import DataLoader, random_split
from src.data.dataset import GWDataset
from src.models.normalizing_flow import RealNVP
from src.training.trainer import train


def main():
    parser = argparse.ArgumentParser(description="Training a Normalizing Flow for Gravitational Wave Signal Modeling")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for optimizer")
    parser.add_argument("--input_dim", type=int, default=4096, help="Dimensionality of input signals (length of each signal)")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Dimensionality of hidden layers")
    parser.add_argument("--conditioning_dim", type=int, default=2, help="Dimensionality of conditioning input (e.g., masses)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--f_lower", type=float, default=10.0, help="Lower frequency cutoff for waveform")
    parser.add_argument("--sample_rate", type=int, default=2048, help="Sampling rate for waveform generation")
    parser.add_argument("--duration", type=int, default=2, help="Duration of the waveform signal (seconds)")
    args = parser.parse_args()

    print("Initializing dataset...")
    dataset = GWDataset(
        num_samples=128,
        m1_range=(10, 50),
        m2_range=(10, 50),
        f_lower=args.f_lower,
        duration=args.duration,
        sample_rate=args.sample_rate,
        input_dim=args.input_dim
    )
    print("Dataset initialized")

    print("Splitting dataset into training and validation sets...")
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    print(f"Training size: {train_size}, Validation size: {val_size}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    print("DataLoader initialized")

    print("Initializing model...")
    model = RealNVP(input_dim=args.input_dim, hidden_dim=args.hidden_dim, conditioning_dim=args.conditioning_dim)
    model.to(args.device)
    print("Model initialized")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print("Optimizer initialized")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)

    print(f"Training on {args.device} with input_dim={args.input_dim}, hidden_dim={args.hidden_dim}, batch_size={args.batch_size}")
    train(model, train_loader, val_loader, optimizer, scheduler, num_epochs=args.num_epochs, device=args.device)

    print("Saving the trained model...")
    torch.save(model.state_dict(), "realnvp_model.pth")
    print("Model saved")


if __name__ == "__main__":
    main()
