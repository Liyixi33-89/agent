import gradio as gr
import torch
import pandas as pd
import numpy as np
from transformers import pipeline
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class GradioDemo:
    """Gradio 演示界面"""
    
    def __init__(self):
        self.models = {}
        self.loaded_model = None
        self.loaded_tokenizer = None
    
    def load_model(self, model_path: str):
        """加载模型"""
        try:
            from modeling_bert import load_saved_model, load_tokenizer
            self.loaded_model = load_saved_model(model_path)
            self.loaded_tokenizer = load_tokenizer()
            return f"模型加载成功: {model_path}"
        except Exception as e:
            return f"模型加载失败: {str(e)}"
    
    def predict_text(self, text: str) -> Dict[str, Any]:
        """文本分类预测"""
        if not self.loaded_model or not self.loaded_tokenizer:
            return {"error": "请先加载模型"}
        
        try:
            from trainer import Trainer
            trainer = Trainer(self.loaded_model)
            predictions = trainer.predict([text], self.loaded_tokenizer)
            
            # 获取预测概率
            encoding = self.loaded_tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.loaded_model(
                    input_ids=encoding['input_ids'].to(self.loaded_model.device),
                    attention_mask=encoding['attention_mask'].to(self.loaded_model.device)
                )
                probabilities = torch.softmax(outputs['logits'], dim=1)
                probabilities = probabilities.cpu().numpy()[0]
            
            return {
                "prediction": int(predictions[0]),
                "probabilities": probabilities.tolist(),
                "confidence": float(np.max(probabilities))
            }
        except Exception as e:
            return {"error": f"预测失败: {str(e)}"}
    
    def create_training_dashboard(self, history: Dict[str, Any]) -> plt.Figure:
        """创建训练过程可视化图表"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 损失曲线
        axes[0, 0].plot(history['train_loss'], label='训练损失')
        axes[0, 0].plot(history['val_loss'], label='验证损失')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # 准确率曲线
        axes[0, 1].plot(history['train_accuracy'], label='训练准确率')
        axes[0, 1].plot(history['val_accuracy'], label='验证准确率')
        axes[0, 1].set_title('准确率曲线')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # F1分数曲线
        axes[1, 0].plot(history['val_f1'], label='验证F1分数', color='green')
        axes[1, 0].set_title('F1分数曲线')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        
        # 混淆矩阵（示例）
        # 这里可以添加混淆矩阵可视化
        axes[1, 1].text(0.5, 0.5, '混淆矩阵可视化', ha='center', va='center', fontsize=16)
        axes[1, 1].set_title('混淆矩阵')
        
        plt.tight_layout()
        return fig

def create_gradio_interface():
    """创建 Gradio 界面"""
    demo = GradioDemo()
    
    with gr.Blocks(title="Agent Finetune Demo", theme="soft") as interface:
        gr.Markdown("# 🤖 Agent 微调演示平台")
        gr.Markdown("基于 Transformers 的文本分类模型微调演示")
        
        with gr.Tab("模型预测"):
            with gr.Row():
                with gr.Column():
                    model_path_input = gr.Textbox(
                        label="模型路径",
                        placeholder="输入模型文件路径（如：models/my_model.pth）",
                        value="models/sample_model.pth"
                    )
                    load_model_btn = gr.Button("加载模型")
                    load_status = gr.Textbox(label="加载状态", interactive=False)
                
                with gr.Column():
                    text_input = gr.Textbox(
                        label="输入文本",
                        placeholder="请输入要分类的文本...",
                        lines=3
                    )
                    predict_btn = gr.Button("预测")
            
            with gr.Row():
                prediction_output = gr.JSON(label="预测结果")
                
            # 绑定事件
            load_model_btn.click(
                fn=demo.load_model,
                inputs=model_path_input,
                outputs=load_status
            )
            predict_btn.click(
                fn=demo.predict_text,
                inputs=text_input,
                outputs=prediction_output
            )
        
        with gr.Tab("训练可视化"):
            gr.Markdown("## 训练过程可视化")
            
            # 示例历史数据
            example_history = {
                'train_loss': [0.8, 0.6, 0.4, 0.3, 0.25],
                'val_loss': [0.9, 0.7, 0.5, 0.4, 0.35],
                'train_accuracy': [0.65, 0.75, 0.82, 0.88, 0.92],
                'val_accuracy': [0.62, 0.72, 0.78, 0.84, 0.88],
                'val_f1': [0.61, 0.71, 0.77, 0.83, 0.87]
            }
            
            plot_output = gr.Plot(label="训练过程图表")
            
            # 显示示例图表
            interface.load(
                fn=lambda: demo.create_training_dashboard(example_history),
                outputs=plot_output
            )
        
        with gr.Tab("模型信息"):
            gr.Markdown("## 支持的模型类型")
            
            model_info = {
                "BERT-base": "基于BERT的文本分类模型，适用于通用文本分类任务",
                "BERT-large": "更大的BERT模型，适合复杂分类任务",
                "RoBERTa": "优化的BERT变体，在多个NLP任务上表现优异",
                "DistilBERT": "轻量级BERT模型，推理速度快，适合生产环境"
            }
            
            for model_name, description in model_info.items():
                with gr.Group():
                    gr.Markdown(f"### {model_name}")
                    gr.Markdown(f"{description}")
    
    return interface

if __name__ == "__main__":
    # 创建并启动 Gradio 界面
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )