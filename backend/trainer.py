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
import gc

def clear_gpu_memory():
    """清理GPU显存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

class Trainer:
    """模型训练器 - 支持梯度累积和显存优化"""
    
    def __init__(self, model: nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # 如果使用GPU，启用内存优化
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
    
    def train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                   scheduler: Optional[Any] = None, 
                   gradient_accumulation_steps: int = 1,
                   use_amp: bool = False,
                   scaler: Optional[Any] = None) -> Dict[str, float]:
        """训练一个epoch - 支持梯度累积和混合精度训练"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        optimizer.zero_grad()  # 在epoch开始时清零梯度
        
        for batch_idx, batch in enumerate(train_loader):
            # 将数据移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 混合精度训练
            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs['loss'] / gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                # 前向传播
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss'] / gradient_accumulation_steps
                # 反向传播
                loss.backward()
            
            # 统计信息（使用原始loss值）
            total_loss += loss.item() * gradient_accumulation_steps
            
            # 获取预测结果
            with torch.no_grad():
                preds = torch.argmax(outputs['logits'], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            # 梯度累积：每 gradient_accumulation_steps 步更新一次参数
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_amp and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad()
                
                # 定期清理显存
                if (batch_idx + 1) % (gradient_accumulation_steps * 10) == 0:
                    clear_gpu_memory()
        
        # 处理剩余的梯度
        if (batch_idx + 1) % gradient_accumulation_steps != 0:
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
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
        
        # 评估后清理显存
        clear_gpu_memory()
        
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
    
    def predict(self, texts: List[str], tokenizer, max_length: int = 128) -> List[int]:
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
    device: str = None,
    gradient_accumulation_steps: int = 4,  # 新增：梯度累积步数
    use_amp: bool = True  # 新增：是否使用混合精度训练
) -> Dict[str, Any]:
    """
    训练模型 - 支持梯度累积和混合精度训练
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        learning_rate: 学习率
        warmup_steps: 预热步数
        progress_callback: 进度回调函数，参数为 (当前epoch, 总epochs, 进度百分比)
        device: 设备类型，cuda 或 cpu，默认自动检测
        gradient_accumulation_steps: 梯度累积步数，用于减少显存占用
        use_amp: 是否使用混合精度训练（仅GPU有效）
    """
    # 设备选择: 优先使用传入的 device，否则自动检测
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 清理显存后再开始训练
    clear_gpu_memory()
    
    print(f"💻 训练设备: {device.upper()}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   显存: {total_mem:.1f} GB")
        print(f"   梯度累积: {gradient_accumulation_steps} 步 (等效batch_size: {train_loader.batch_size * gradient_accumulation_steps})")
        if use_amp:
            print(f"   混合精度训练: 已启用 (FP16)")
    else:
        use_amp = False  # CPU不支持AMP
    
    trainer = Trainer(model, device)
    
    # 优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # 混合精度训练scaler
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device == "cuda") else None
    
    # 学习率调度器（考虑梯度累积）
    total_steps = (len(train_loader) // gradient_accumulation_steps) * epochs
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
    
    try:
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # 计算当前进度并调用回调
            progress = ((epoch + 0.5) / epochs) * 100  # 训练中间点
            if progress_callback:
                try:
                    progress_callback(epoch + 1, epochs, progress)
                except Exception as e:
                    print(f"Progress callback error: {e}")
            
            # 训练（使用梯度累积和混合精度）
            train_metrics = trainer.train_epoch(
                train_loader, optimizer, scheduler,
                gradient_accumulation_steps=gradient_accumulation_steps,
                use_amp=use_amp,
                scaler=scaler
            )
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
            
            # 显示GPU显存使用情况
            if device == "cuda":
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f"GPU显存: 已分配 {allocated:.2f}GB / 已保留 {reserved:.2f}GB")
            
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
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # 每个epoch后清理显存
            clear_gpu_memory()
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n❌ GPU显存不足！建议:")
            print(f"   1. 减小 batch_size (当前: {train_loader.batch_size})")
            print(f"   2. 减小 max_length")
            print(f"   3. 增大 gradient_accumulation_steps (当前: {gradient_accumulation_steps})")
            clear_gpu_memory()
        raise e
    
    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # 最终清理
    clear_gpu_memory()
    
    return history