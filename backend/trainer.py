import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional, Callable
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import os

class Trainer:
    """模型训练器"""
    
    def __init__(self, model: nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                   scheduler: Optional[Any] = None) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            # 统计信息
            total_loss += loss.item()
            
            # 获取预测结果
            preds = torch.argmax(outputs['logits'], dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """模型评估"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
                
                total_loss += loss.item()
                
                preds = torch.argmax(outputs['logits'], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def predict(self, texts: List[str], tokenizer, max_length: int = 512) -> List[int]:
        """预测"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                # 编码文本
                encoding = tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                # 预测
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pred = torch.argmax(outputs['logits'], dim=1)
                predictions.append(pred.cpu().item())
        
        return predictions

def train_model(
    model: nn.Module, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    epochs: int = 3, 
    learning_rate: float = 2e-5, 
    warmup_steps: int = 0,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    device: str = None  # 新增: 允许指定设备
) -> Dict[str, Any]:
    """
    训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        learning_rate: 学习率
        warmup_steps: 预热步数
        progress_callback: 进度回调函数，参数为 (当前eepoch, 总epochs, 进度百分比)
        device: 设备类型，cuda 或 cpu，默认自动检测
    """
    # 设备选择: 优先使用传入的 device，否则自动检测
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"💻 训练设备: {device.upper()}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    trainer = Trainer(model, device)    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    best_val_f1 = 0
    best_model_state = None
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # 计算当前进度并调用回调
        progress = ((epoch + 0.5) / epochs) * 100  # 训练中间点
        if progress_callback:
            try:
                progress_callback(epoch + 1, epochs, progress)
            except Exception as e:
                print(f"Progress callback error: {e}")
        
        # 训练
        train_metrics = trainer.train_epoch(train_loader, optimizer, scheduler)
        history['train_loss'].append(train_metrics['loss'])
        history['train_accuracy'].append(train_metrics['accuracy'])
        
        # 验证
        val_metrics = trainer.evaluate(val_loader)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1_score'])
        
        print(f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Val F1: {val_metrics['f1_score']:.4f}")
        
        # 每个epoch结束后更新进度
        progress = ((epoch + 1) / epochs) * 100
        if progress_callback:
            try:
                progress_callback(epoch + 1, epochs, progress)
            except Exception as e:
                print(f"Progress callback error: {e}")
        
        # 保存最佳模型
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            best_model_state = model.state_dict()
    
    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return history