# FlowLite

FlowLite is a lightweight deep learning framework designed for educational purposes, with an API similar to PyTorch. It simplifies the process of building, training, and deploying neural networks, making it ideal for beginners and educators in the field of AI.

## Installation

You can install FlowLite using pip:

```bash
pip install flowlite
```


## Quick Start
Here's a quick example of using FlowLite to create a simple neural network:
```python
import flowlite
import flowlite.nn as nn
import flowlite.optim as optim
import flowlite.functional as F

# Define a simple model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
model = Net()

# Define a loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy data for the sake of example
inputs = flowlite.randn(64, 784)  # Batch size 64, input dimension 784
labels = flowlite.randint(0, 10, (64,))  # Random labels for a batch size of 64

# Forward pass
outputs = model(inputs)
loss = loss_fn(outputs, labels)

# Backward pass and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()

print('Training step complete.')
```

## License
This project is licensed under the [Apache License (Version 2.0)](https://github.com/caaatch22/flowlite/blob/main/LICENSE).