import argparse
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from model import MNISTModel
from data import load_data
import torch.nn.functional as F

def train(args):
    model = MNISTModel()
    train_loader, _ = load_data(args.batch_size)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    train_losses = []
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        running_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            this_loss = loss.item()
            running_loss += this_loss
            train_losses.append(this_loss)
        
        print(f"Loss: {running_loss / len(train_loader)}")

    torch.save(model.state_dict(), "trained_model.pt")

    plt.plot(train_losses)
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.title("Training Curve")
    plt.savefig("training_curve.png")

def evaluate(args):
    model = MNISTModel()
    model.load_state_dict(torch.load(args.model_path))
    _, test_loader = load_data(batch_size=1000)

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f"Test set accuracy: {100. * correct / len(test_loader.dataset)}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    subparsers = parser.add_subparsers(dest='mode')
    
    # Training settings
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--batch-size', type=int, default=64)
    parser_train.add_argument('--epochs', type=int, default=10)
    parser_train.add_argument('--lr', type=float, default=1e-4)
    parser_train.set_defaults(func=train)

    # Evaluation settings
    parser_evaluate = subparsers.add_parser('evaluate')
    parser_evaluate.add_argument('--model-path', type=str, required=True)
    parser_evaluate.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)

