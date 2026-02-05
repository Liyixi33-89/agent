import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional, Tuple
from sklearn.model_selection import train_test_split

class TextClassificationDataset(Dataset):
    """文本分类数据集类"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: AutoTokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 对文本进行编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_csv_data(file_path: str, text_column: str = "text", label_column: str = "target") -> Tuple[List[str], List[int]]:
    """从CSV文件加载数据"""
    try:
        df = pd.read_csv(file_path)
        texts = df[text_column].tolist()
        labels = df[label_column].tolist()
        return texts, labels
    except Exception as e:
        raise ValueError(f"Failed to load data from {file_path}: {str(e)}")

def load_json_data(file_path: str, text_key: str = "text", label_key: str = "label") -> Tuple[List[str], List[int]]:
    """从JSON文件加载数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [item[text_key] for item in data]
        labels = [item[label_key] for item in data]
        return texts, labels
    except Exception as e:
        raise ValueError(f"Failed to load data from {file_path}: {str(e)}")

def create_data_loader(texts: List[str], labels: List[int], tokenizer: AutoTokenizer, 
                      batch_size: int = 32, max_length: int = 512, shuffle: bool = False) -> DataLoader:
    """创建数据加载器"""
    dataset = TextClassificationDataset(texts, labels, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def split_data(texts: List[str], labels: List[int], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple:
    """划分训练集、验证集和测试集
    
    当数据量太少时会自动取消分层采样，避免报错
    """
    from collections import Counter
    
    # 检查每个类别的数量
    label_counts = Counter(labels)
    min_count = min(label_counts.values())
    
    # 如果最小类别数量小于3，则不使用分层采样
    use_stratify = min_count >= 3
    stratify_param = labels if use_stratify else None
    
    if not use_stratify:
        print(f"⚠️ 警告: 某些类别数据量太少 (最小: {min_count})，已禁用分层采样")
    
    # 先划分训练集和临时集
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, labels, train_size=train_ratio, random_state=42, stratify=stratify_param
    )
    
    # 检查临时集中每个类别的数量
    temp_label_counts = Counter(temp_labels)
    temp_min_count = min(temp_label_counts.values())
    temp_stratify_param = temp_labels if temp_min_count >= 2 else None
    
    # 再从临时集划分验证集和测试集
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, train_size=val_ratio_adjusted, random_state=42, stratify=temp_stratify_param
    )
    
    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels