import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score


class MLPClassifier_for_bank(nn.Module):
    def __init__(self, input_size, hidden_sizes=[32, 16, 8], dropout_rate=0.3):
        super(MLPClassifier_for_bank, self).__init__()
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
    # Загрузка параметров
    with open('params.yaml', 'r') as f:
        import yaml
        params = yaml.safe_load(f)

    hidden_sizes = params.get('hidden_sizes_bank', [32, 16, 8])
    dropout_rate = params.get('dropout_rate_bank', 0.3)
    batch_size = params.get('batch_size_bank', 64)
    num_epochs = params.get('num_epochs_bank', 30)
    learning_rate = params.get('learning_rate_bank', 0.001)
    step_size = params.get('step_size_bank', 30)
    gamma = params.get('gamma_bank', 0.1)

    print(f"🔧 Параметры обучения (С НУЛЯ):")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Эпох: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Hidden sizes: {hidden_sizes}")
    print("-" * 50)

    # Загрузка данных Bank
    X_train = np.load('data/processed/bank/X_train.npy')
    X_test = np.load('data/processed/bank/X_test.npy')
    y_train = np.load('data/processed/bank/y_train.npy')
    y_test = np.load('data/processed/bank/y_test.npy')

    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")

    # Тензоры
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)

    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Создание модели
    model_bank = MLPClassifier_for_bank(
        input_size=X_train.shape[1],
        hidden_sizes=hidden_sizes,
        dropout_rate=dropout_rate
    )

    print(f"\n✅ Модель создана (обучение с нуля)")
    print(model_bank)

    # Оптимизатор
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model_bank.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Обучение
    model_bank.train()
    train_losses = []

    print(f"\n🚀 Начинаем обучение с нуля...\n")

    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model_bank(batch_X)
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

    # Оценка
    model_bank.eval()

    with torch.no_grad():
        train_pred = model_bank(X_train_tensor)
        train_proba = train_pred.numpy().flatten()
        train_classes = (train_pred >= 0.5).float().numpy().flatten()

        test_pred = model_bank(X_test_tensor)
        test_proba = test_pred.numpy().flatten()
        test_classes = (test_pred >= 0.5).float().numpy().flatten()

    # Метрики
    metrics = {
        'train_accuracy': float(accuracy_score(y_train, train_classes)),
        'train_auc': float(roc_auc_score(y_train, train_proba)),
        'test_accuracy': float(accuracy_score(y_test, test_classes)),
        'test_auc': float(roc_auc_score(y_test, test_proba)),
        'test_precision': float(precision_score(y_test, test_classes)),
        'test_recall': float(recall_score(y_test, test_classes)),
        'test_f1': float(f1_score(y_test, test_classes)),
        'training_type': 'from_scratch'
    }

    print("\n📊 МЕТРИКИ НА ТЕСТЕ:")
    print("-" * 30)
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name:15s}: {value:.4f}")
        else:
            print(f"{name:15s}: {value}")
    print("-" * 30)

    # Сохранение модели
    os.makedirs('models', exist_ok=True)
    torch.save(model_bank.state_dict(), 'models/bank_from_scratch.pth')
    print("\n✓ Модель сохранена!")

    # Сохранение метрик
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/bank_scratch_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✓ Метрики сохранены!")


if __name__ == "__main__":
    main()