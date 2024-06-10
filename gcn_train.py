import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import sys
import torch.nn.functional as F
sys.stdout.reconfigure(encoding='utf-8')


# Load data
with open('C:\\Users\\Lenovo\\martial_art_movement_recongnition\\6m2d\\all_keypoints.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Label encoding
labels = [item['label'] for item in data]
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
# 保存 LabelEncoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Assuming each action has the same number of keypoints
num_keypoints = len(data[0]['keypoints'])

# Create graph data
def create_graph(data, encoded_labels):
    edge_index = []
    for i in range(num_keypoints - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    
    graphs = []
    for idx, item in enumerate(data):
        keypoints = torch.tensor(item['keypoints'], dtype=torch.float)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        label = torch.tensor(encoded_labels[idx], dtype=torch.long).unsqueeze(0)
        graphs.append(Data(x=keypoints, edge_index=edge_index_tensor, y=label))
    
    return graphs

graphs = create_graph(data, encoded_labels)

# Split dataset
train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
train_graphs, val_graphs = train_test_split(train_graphs, test_size=0.2, random_state=42)

# Create data loaders
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)  # Smaller batch size
val_loader = DataLoader(val_graphs, batch_size=32)
test_loader = DataLoader(test_graphs, batch_size=32)

# Model definition
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)  # Adding an additional GCN layer
        self.bn4 = BatchNorm(hidden_dim)
        self.conv5 = GCNConv(hidden_dim, hidden_dim)  # Adding another additional GCN layer
        self.bn5 = BatchNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.5)  # Adjust Dropout rate to 0.5
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.conv5(x, edge_index)
        x = self.bn5(x)
        x = torch.relu(x)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Parameters
input_dim = 3  # Dimension of keypoints
hidden_dim = 512  # Keep the hidden dimension
output_dim = 5  # Number of action classes
model = GCN(input_dim, hidden_dim, output_dim)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # Use AdamW optimizer and adjust learning rate
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            out = model(data)
            loss = criterion(out, data.y.view(-1))
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == data.y.view(-1)).sum().item()
    accuracy = correct / len(loader.dataset)
    return total_loss / len(loader), accuracy

# Training loop with early stopping
num_epochs = 200
early_stopping_patience = 20
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss, val_accuracy = evaluate(model, val_loader, criterion)
    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    scheduler.step()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print('Early stopping')
        break

# Load the best model and test
model.load_state_dict(torch.load('best_model.pt'))
test_loss, test_accuracy = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# Save the model with accuracy in the filename if accuracy > 90.0%
if test_accuracy > 0.88:
    accuracy_str = f"{test_accuracy:.4f}".replace('.', '_')
    model_filename = f'gcn_model_{accuracy_str}.pth'
    torch.save(model.state_dict(), model_filename)
    print(f'Model saved to {model_filename}')
else:
    print(f'Model not saved, accuracy {test_accuracy:.4f} did not exceed 90.0%')
