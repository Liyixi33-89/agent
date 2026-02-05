import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoTokenizer
from typing import Optional, Dict, Any

class BertForTextClassification(nn.Module):
    """基于BERT的文本分类模型"""
    
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 2, dropout_rate: float = 0.1):
        super(BertForTextClassification, self).__init__()
        
        # 加载BERT模型配置
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels
        
        # 加载BERT模型
        self.bert = AutoModel.from_pretrained(model_name, config=self.config)
        
        # 分类器
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """前向传播"""
        
        # 获取BERT输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 获取[CLS]标记的隐藏状态
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # 应用dropout
        pooled_output = self.dropout(pooled_output)
        
        # 分类
        logits = self.classifier(pooled_output)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions
        }

def load_model(model_name: str, num_labels: int = 2, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> BertForTextClassification:
    """加载模型"""
    model = BertForTextClassification(model_name, num_labels)
    model.to(device)
    return model

def load_tokenizer(model_name: str = "bert-base-uncased") -> AutoTokenizer:
    """加载分词器"""
    return AutoTokenizer.from_pretrained(model_name)

def save_model(model: BertForTextClassification, save_path: str):
    """保存模型"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model.config,
        'num_labels': model.config.num_labels
    }, save_path)

def load_saved_model(model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> BertForTextClassification:
    """加载已保存的模型"""
    # 添加 weights_only=False 以兼容 PyTorch 2.6+
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 创建模型
    model = BertForTextClassification(
        model_name=checkpoint['config']._name_or_path,
        num_labels=checkpoint['num_labels']
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model