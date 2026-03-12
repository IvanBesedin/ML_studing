import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score


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


def freeze_layers(model, num_linear_to_freeze=0, freeze_bn=False):
    linear_count = 0
    total_linear = 0
    
    for module in model.networks:
        if isinstance(module, nn.Linear):
            total_linear += 1
    
    print(f"📊 Всего Linear слоёв: {total_linear}")
    print(f"🔒 Планируем заморозить: {num_linear_to_freeze} Linear" + 
          (f" + BatchNorm" if freeze_bn else ""))
    print("-" * 50)
    
    for i, module in enumerate(model.networks):
        if isinstance(module, nn.Linear):
            if linear_count < num_linear_to_freeze:
                module.weight.requires_grad = False
                module.bias.requires_grad = False
                print(f"🔒 Linear #{linear_count+1}: {module.in_features}→{module.out_features}")
            else:
                module.weight.requires_grad = True
                module.bias.requires_grad = True
                print(f"🔓 Linear #{linear_count+1}: {module.in_features}→{module.out_features}")
            linear_count += 1
        
        elif isinstance(module, nn.BatchNorm1d):
            if freeze_bn and linear_count <= num_linear_to_freeze:
                module.weight.requires_grad = False
                module.bias.requires_grad = False
                module.eval()
                print(f"🔒 BatchNorm (после Linear #{linear_count})")
            else:
                module.weight.requires_grad = True
                module.bias.requires_grad = True
                module.train()
                print(f"🔓 BatchNorm (после Linear #{linear_count})")
    
    print("-" * 50)
    print(f"✅ Заморожено: {num_linear_to_freeze}/{total_linear} Linear слоёв")
    if freeze_bn:
        print(f"✅ BatchNorm заморожен вместе с Linear")
    else:
        print(f"✅ BatchNorm остаётся обучаемым")


def main():
    # Загрузка параметров
    with open('params.yaml', 'r') as f:
        import yaml
        params = yaml.safe_load(f)
    
    # Загрузка данных Bank
    X_train = np.load('data/processed/bank/X_train.npy')
    X_test = np.load('data/processed/bank/X_test.npy')
    y_train = np.load('data/processed/bank/y_train.npy')
    y_test = np.load('data/processed/bank/y_test.npy')
    
    # Тензоры
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
    
    # DataLoader
    batch_size = params.get('batch_size', 64)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Создание модели
    model = MLPClassifier(
        input_size=X_train.shape[1],
        hidden_sizes=params.get('hidden_sizes', [64, 32, 16]),
        dropout_rate=params.get('dropout_rate', 0.3)
    )
    
    # Загрузка предобученных весов
    pretrained_path = 'models/adult_pretrained.pth'
    
    try:
        pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cpu'), weights_only=True)
        model_dict = model.state_dict()
        
        if isinstance(pretrained_dict, dict) and 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                           if k in model_dict and v.shape == model_dict[k].shape}
        
        missing = [k for k in model_dict.keys() if k not in pretrained_dict]
        if missing:
            print(f"⚠ Отсутствующие ключи: {missing[:5]}")
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        
        print(f"✅ Веса успешно загружены из {pretrained_path}")
        print(f"📦 Загружено {len(pretrained_dict)} параметра")
        
    except FileNotFoundError:
        print(f"⚠ Файл {pretrained_path} не найден, обучаем с нуля!")
    except Exception as e:
        print(f"⚠ Ошибка при загрузке весов: {e}")
    
    # Заморозка слоёв
    num_frozen = params.get('num_frozen_layers', 1)
    freeze_bn = params.get('freeze_bn', True)
    
    if num_frozen > 0:
        freeze_layers(model, num_linear_to_freeze=num_frozen, freeze_bn=freeze_bn)
    
    # Оптимизатор
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=params.get('fine_tuning_learning_rate', 0.0005)
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=params.get('fine_tuning_step_size', 30),
        gamma=params.get('fine_tuning_gamma', 0.1)
    )
    
    # Обучение
    model.train()
    num_epochs = params.get('fine_tuning_num_epochs', 30)
    train_losses = []
    
    print("\n🚀 Начинаем дообучение (Fine-Tuning)...\n")
    
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
            print(f'Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f} | Acc = {accuracy:.4f}')
    
    # Оценка
    model.eval()
    
    with torch.no_grad():
        train_pred = model(X_train_tensor)
        train_proba = train_pred.numpy().flatten()
        train_classes = (train_pred >= 0.5).float().numpy().flatten()
        
        test_pred = model(X_test_tensor)
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
        'test_f1': float(f1_score(y_test, test_classes))
    }
    
    print("\n📊 Метрики на тесте:")
    print("-" * 30)
    for name, value in metrics.items():
        print(f"{name:15s}: {value:.4f}")
    print("-" * 30)
    
    # Сохранение модели
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/bank_finetuned.pth')
    print("\n✓ Модель сохранена!")
    
    # Сохранение метрик
    os.makedirs('metrics', exist_ok=True)
    with open('metrics/bank_finetune_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✓ Метрики сохранены!")


if __name__ == "__main__":
    main()