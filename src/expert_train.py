import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from simrocketenv import SimRocketEnv
from nnpolicynetwork import NNPolicyNetwork

def train_and_evaluate():

    # Load data from JSON
    with open("expert_data.json", "r") as file:
        data = json.load(file)

    # Assuming each entry in data is a dictionary with 'obs' and 'acts' keys
    observations = np.array([item['obs'] for item in data])
    actions = np.array([item['acts'] for item in data])

    # pytorch setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch: using {device}")

    observations = torch.tensor(observations, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    dataset = TensorDataset(observations, actions)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    input_size = observations.shape[1]
    output_size = actions.shape[1]
    model = NNPolicyNetwork(input_size, output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for inputs, targets in data_loader:
            # Move data to the GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Compute the average loss for this epoch
        avg_loss = running_loss / len(data_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    torch.save(model, 'torch_nn_mpc-rocket-v0.pth')

    # Inference example:

    # model = torch.load('torch_nn_mpc-rocket-v0.pth')
    # model.eval()  # Set the model to inference mode
    # state_vector = ... input state vector
    # state_tensor = torch.tensor(state_vector, dtype=torch.float32)
    # state_tensor = state_tensor.unsqueeze(0)  # Not sure if req.: Adding batch dimension
    # # Run inference
    # with torch.no_grad():  # Disables gradient calculation, which is not needed during inference
    #     action_pred, state_pred = model(state_tensor)
    # # Convert predictions to numpy arrays
    # action_pred = action_pred.numpy()
    # state_pred = state_pred.numpy()

if __name__ == '__main__':
    train_and_evaluate()

