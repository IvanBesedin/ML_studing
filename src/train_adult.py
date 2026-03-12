import torch as tch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
from sklearn.metrics import accuracy_score, roc_auc_score

# === Модель MLP ===
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 32, 16], dropout_rate=0.3):
        super(MLPClassifier, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.networks = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.networks(x)

def main():
    # === Загрузка параметров ===
    with open('params.yaml', 'r') as f:
        import yaml
        params = yaml.safe_load(f)
    
    # === Загрузка данных ===
    X_train = np.load('data/processed/adult/X_train.npy')
    X_test = np.load('data/processed/adult/X_test.npy')
    y_train = np.load('data/processed/adult/y_train.npy')
    y_test = np.load('data/processed/adult/y_test.npy')
    
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    
    # === Тензоры ===
    X_train_tensor = tch.FloatTensor(X_train)
    y_train_tensor = tch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = tch.FloatTensor(X_test)
    y_test_tensor = tch.FloatTensor(y_test).reshape(-1, 1)
    
    # === DataLoader ===
    batch_size = params.get('batch_size', 64)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # === Модель ===
    model = MLPClassifier(
        input_size=X_train.shape[1],
        hidden_sizes=params.get('hidden_sizes', [64, 32, 16]),
        dropout_rate=params.get('dropout_rate', 0.3)
    )
    
    # === Loss и оптимизатор ===
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params.get('learning_rate', 0.001))
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=params.get('step_size', 30), 
        gamma=params.get('gamma', 0.1)
    )
    
    # === Обучение ===
    model.train()
    num_epochs = params.get('num_epochs', 50)
    train_losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            predicted_classes = (outputs >= 0.5).float()
            correct += (predicted_classes == batch_y).sum().item()
            total += batch_y.size(0)
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        accuracy = correct / total
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f} | Accuracy = {accuracy:.4f}')
    
    # === Оценка на тесте ===
    model.eval()
    with tch.no_grad():
        # Train
        train_pred = model(X_train_tensor)
        train_proba = train_pred.numpy().flatten()
        train_classes = (train_pred >= 0.5).float().numpy().flatten()
        
        train_acc = accuracy_score(y_train, train_classes)
        train_auc = roc_auc_score(y_train, train_proba)
        
        # Test
        test_pred = model(X_test_tensor)
        test_proba = test_pred.numpy().flatten()
        test_classes = (test_pred >= 0.5).float().numpy().flatten()
        
        test_acc = accuracy_score(y_test, test_classes)
        test_auc = roc_auc_score(y_test, test_proba)
    
    # === Вывод метрик ===
    print("\n📊 Сравнение метрик:")
    print(f"{'Метрика':<12} {'Train':>10} {'Test':>10} {'Разница':>10}")
    print("-" * 45)
    print(f"{'Accuracy':<12} {train_acc:>10.4f} {test_acc:>10.4f} {abs(train_acc-test_acc):>10.4f}")
    print(f"{'ROC-AUC':<12} {train_auc:>10.4f} {test_auc:>10.4f} {abs(train_auc-test_auc):>10.4f}")
    
    # === Сохранение модели ===
    os.makedirs('models', exist_ok=True)
    tch.save(model.state_dict(), 'models/adult_pretrained.pth')
    print("\n✓ Модель сохранена!")
    
    # === Сохранение метрик ===
    os.makedirs('metrics', exist_ok=True)
    metrics = {
        'train_accuracy': float(train_acc),
        'train_auc': float(train_auc),
        'test_accuracy': float(test_acc),
        'test_auc': float(test_auc),
        'accuracy_diff': float(abs(train_acc - test_acc)),
        'auc_diff': float(abs(train_auc - test_auc))
    }
    
    with open('metrics/adult_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✓ Метрики сохранены!")

if __name__ == "__main__":
    main()