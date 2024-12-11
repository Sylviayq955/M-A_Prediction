import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import traceback

def clean_financial_data(df, features):
    """清理财务数据"""
    df_clean = df[features].copy()
    
    # 替换无穷值
    df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 使用中位数填充缺失值
    medians = df_clean.median()
    df_clean.fillna(medians, inplace=True)
    
    # 处理异常值：将超过3个标准差的值设为3个标准差的值
    for col in df_clean.columns:
        mean = df_clean[col].mean()
        std = df_clean[col].std()
        df_clean[col] = df_clean[col].clip(mean - 3*std, mean + 3*std)
    
    return df_clean

class STMOTE:
    """Structure-based SMOTE算法实现"""
    def __init__(self, k_neighbors=5, random_state=42, structural_weight=0.5):
        self.k = k_neighbors
        self.random_state = random_state
        self.structural_weight = structural_weight
    
    def fit_resample(self, X, y):
        try:
            X = torch.FloatTensor(X)
            y = torch.LongTensor(y)
            
            # 确保X没有无穷值或NaN
            if torch.isnan(X).any() or torch.isinf(X).any():
                raise ValueError("输入数据包含无穷值或NaN")
            
            minority_class = 0 if (y == 0).sum() < (y == 1).sum() else 1
            minority_indices = (y == minority_class).nonzero().squeeze()
            
            # 处理只有一个少数类样本的情况
            if minority_indices.dim() == 0:
                minority_indices = minority_indices.unsqueeze(0)
            
            majority_indices = (y != minority_class).nonzero().squeeze()
            
            n_minority = len(minority_indices)
            n_majority = len(majority_indices)
            n_synthetic = n_majority - n_minority
            
            if n_synthetic <= 0:
                return X.numpy(), y.numpy()
            
            minority_samples = X[minority_indices]
            
            # 计算相似度矩阵
            norm = torch.norm(minority_samples, dim=1, keepdim=True)
            normalized_samples = minority_samples / (norm + 1e-8)
            similarity = torch.mm(normalized_samples, normalized_samples.t())
            
            # 找到最近邻
            k = min(self.k + 1, len(minority_indices))
            _, nearest_neighbors = torch.topk(similarity, k, dim=1)
            
            synthetic_samples = []
            np.random.seed(self.random_state)
            
            for i in range(n_minority):
                n_synthetic_i = int(np.ceil(n_synthetic / n_minority))
                current_sample = minority_samples[i]
                
                for _ in range(n_synthetic_i):
                    if k > 1:
                        nn_idx = np.random.randint(1, k)
                        nn = minority_samples[nearest_neighbors[i, nn_idx]]
                        
                        # 生成合成样本
                        diff_vector = nn - current_sample
                        random_weight = torch.rand(1).item()
                        synthetic_sample = current_sample + random_weight * diff_vector
                        
                        synthetic_samples.append(synthetic_sample)
                        
                        if len(synthetic_samples) >= n_synthetic:
                            break
                
                if len(synthetic_samples) >= n_synthetic:
                    break
            
            if not synthetic_samples:
                return X.numpy(), y.numpy()
            
            synthetic_samples = torch.stack(synthetic_samples[:n_synthetic])
            synthetic_labels = torch.full((n_synthetic,), minority_class, dtype=torch.long)
            
            X_resampled = torch.cat([X, synthetic_samples])
            y_resampled = torch.cat([y, synthetic_labels])
            
            return X_resampled.numpy(), y_resampled.numpy()
            
        except Exception as e:
            print(f"STMOTE处理时出错: {str(e)}")
            traceback.print_exc()
            raise

class MergerPredictor(nn.Module):
    """并购预测神经网络模型"""
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.parameters())
    
    def forward(self, x):
        return self.model(x)

def evaluate_metrics(model, test_loader, device):
    """计算MSE和MAE"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            
            outputs = model(X).squeeze()
            
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    
    return mse, mae

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 1. 加载数据
        df = pd.read_csv('lseg_data/20130101_20130131.csv')
        print(f"数据集大小: {df.shape}")
        
        # 2. 选择特征
        financial_features = [
            # 1. 交易相关指标
            'Rank Value inc. Net Debt of Target',  # 并购交易估值
            
            # 2. 买方财务状况
            'Acquiror Market Value 4 Weeks Prior to Announcement',  # 买方市值
            'Acquiror Total Assets Last 12 Months',  # 买方总资产
            'Acquiror Net Debt Last 12 Months',  # 买方净债务
            'Acquiror EBIT Last 12 Months',  # 买方息税前利润
            'Acquiror EBITDA Last 12 Months',  # 买方EBITDA
            'Acquiror Current Assets Last 12 Months',  # 买方流动资产
            'Acquiror Current Liabilities Last 12 Months',  # 买方流动负债
            
            # 3. 卖方财务状况
            'Target Market Value 4 Weeks Prior to Announcement',  # 卖方市值
            'Total Assets Last 12 Months',  # 卖方总资产
            'Net Debt Last 12 Months',  # 卖方净债务
            'EBIT Last 12 Months',  # 卖方息税前利润
            'EBITDA Last 12 Months',  # 卖方EBITDA
            'Current Assets Last 12 Months',  # 卖方流动资产
            'Current Liabilities Last 12 Months'  # 卖方流动负债
        ]
        
        def calculate_financial_ratios(df):
            """计算关键财务比率"""
            ratios = pd.DataFrame(index=df.index)
            
            try:
                # 1. 买方财务比率
                ratios['acquiror_current_ratio'] = (df['Acquiror Current Assets Last 12 Months'] / 
                                                  df['Acquiror Current Liabilities Last 12 Months'])
                ratios['acquiror_debt_to_asset'] = (df['Acquiror Net Debt Last 12 Months'] / 
                                                   df['Acquiror Total Assets Last 12 Months'])
                ratios['acquiror_profit_margin'] = (df['Acquiror EBIT Last 12 Months'] / 
                                                  df['Acquiror Total Assets Last 12 Months'])
                
                # 2. 卖方财务比率
                ratios['target_current_ratio'] = (df['Current Assets Last 12 Months'] / 
                                                df['Current Liabilities Last 12 Months'])
                ratios['target_debt_to_asset'] = (df['Net Debt Last 12 Months'] / 
                                                df['Total Assets Last 12 Months'])
                ratios['target_profit_margin'] = (df['EBIT Last 12 Months'] / 
                                               df['Total Assets Last 12 Months'])
                
                # 3. 买卖双方对比比率
                ratios['size_ratio'] = (df['Target Market Value 4 Weeks Prior to Announcement'] / 
                                      df['Acquiror Market Value 4 Weeks Prior to Announcement'])
                ratios['relative_profitability'] = (ratios['target_profit_margin'] / 
                                                  ratios['acquiror_profit_margin'])
                
            except Exception as e:
                print(f"计算财务比率时出错: {e}")
            
            # 处理无穷值和NaN
            ratios = ratios.replace([np.inf, -np.inf], np.nan)
            ratios = ratios.fillna(ratios.mean())
            return ratios
        
        # 检查可用特征
        available_features = [f for f in financial_features if f in df.columns]
        if not available_features:
            raise ValueError("没有找到任何可用的财务特征！")
        
        print(f"使用的特征数量: {len(available_features)}")
        print("使用的特征:", available_features)
        
        # 3. 数据清理和准备
        features = clean_financial_data(df, available_features)
        
        # 计算并添加财务比率
        financial_ratios = calculate_financial_ratios(df)
        features = pd.concat([features, financial_ratios], axis=1)
        
        # 准备目标变量
        target = (df['Deal Status'] == 'Completed').astype(int).values
        features = features.fillna(0).values  # 处理可能的缺失值
        
        print("\n目标变量分布:")
        print("完成的并购交易:", np.sum(target == 1))
        print("未完成的并购交易:", np.sum(target == 0))
        
        # 4. 数据集分割
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, stratify=target
        )
        
        # 5. 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 6. STMOTE处理不平衡
        stmote = STMOTE()
        X_train_balanced, y_train_balanced = stmote.fit_resample(X_train_scaled, y_train)
        print(f"\nSTMOTE后的训练集大小: {X_train_balanced.shape}")
        print(f"类别分布: {np.bincount(y_train_balanced)}")
        
        # 7. PCA降维
        pca = PCA(n_components=0.95)
        X_train_pca = pca.fit_transform(X_train_balanced)
        X_test_pca = pca.transform(X_test_scaled)
        
        print(f"\nPCA后的特征数量: {X_train_pca.shape[1]}")
        print(f"累计解释方差比: {sum(pca.explained_variance_ratio_):.4f}")
        
        # 8. 创建数据加载器
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train_pca), 
            torch.FloatTensor(y_train_balanced)  # 改为FloatTensor
        )
        test_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_test_pca), 
            torch.FloatTensor(y_test)  # 改为FloatTensor
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)
        
        # 9. 创建和训练模型
        model = MergerPredictor(input_dim=X_train_pca.shape[1]).to(device)
        print("\n模型结构:", model)
        
        # 10. 训练模型
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'mse': [], 'mae': []}
        best_val_loss = float('inf')
        patience = 10
        counter = 0
        epochs = 100
        
        try:
            for epoch in range(epochs):
                # 训练
                model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    
                    model.optimizer.zero_grad()
                    outputs = model(batch_X).squeeze()
                    loss = model.criterion(outputs, batch_y)
                    
                    loss.backward()
                    model.optimizer.step()
                    
                    train_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    train_correct += (predicted == batch_y).sum().item()
                    train_total += batch_y.size(0)
                
                # 验证
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        batch_X = batch_X.to(device)
                        batch_y = batch_y.to(device)
                        
                        outputs = model(batch_X).squeeze()
                        loss = model.criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                        predicted = (outputs > 0.5).float()
                        val_correct += (predicted == batch_y).sum().item()
                        val_total += batch_y.size(0)
                
                # 计算平均损失和准确率
                train_loss = train_loss / len(train_loader)
                train_acc = train_correct / train_total
                val_loss = val_loss / len(test_loader)
                val_acc = val_correct / val_total
                
                # 记录历史
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
                print(f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
                
                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print("Early stopping triggered")
                        break
                        
                # 在每个epoch结束时计算MSE和MAE
                mse, mae = evaluate_metrics(model, test_loader, device)
                print(f'MSE: {mse:.4f}, MAE: {mae:.4f}')
                
                # 记录到history中
                history['mse'] = mse
                history['mae'] = mae
                
        except Exception as e:
            print(f"训练过程中出错: {str(e)}")
            traceback.print_exc()
            return
        
        # 训练结束后显示最终指标
        final_mse, final_mae = evaluate_metrics(model, test_loader, device)
        print("\n最终评估结果:")
        print(f"MSE: {final_mse:.4f}")
        print(f"MAE: {final_mae:.4f}")
        
        # 绘制结果时添加MSE和MAE的图
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Validation')
        plt.title('Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(history['mse'], label='MSE')
        plt.plot(history['mae'], label='MAE')
        plt.title('Error Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # 收集实验结果
        results = {
            'class_distribution': np.bincount(target),
            'features': available_features,
            'history': {
                'train_loss': history['train_loss'],
                'val_loss': history['val_loss'],
                'train_acc': history['train_acc'],
                'val_acc': history['val_acc']
            },
            'metrics': {
                'mse': final_mse,
                'mae': final_mae
            }
        }
        
        return results
        
    except Exception as e:
        print(f"错误: {str(e)}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main() 