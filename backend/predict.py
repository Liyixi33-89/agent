#!/usr/bin/env python
"""
模型预测命令行工具

使用方法:
    # 单条文本预测
    python predict.py --model models/my_model.pth --text "这是一条测试文本"
    
    # 批量预测（从文件读取）
    python predict.py --model models/my_model.pth --input texts.txt --output results.csv
    
    # 交互式预测
    python predict.py --model models/my_model.pth --interactive
    
    # 列出所有可用模型
    python predict.py --list
"""

import argparse
import os
import sys
import torch
from typing import List, Tuple

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modeling_bert import load_saved_model, load_tokenizer


def list_available_models(models_dir: str = "models") -> List[dict]:
    """列出所有可用的模型"""
    if not os.path.exists(models_dir):
        return []
    
    models = []
    for filename in os.listdir(models_dir):
        if filename.endswith('.pth'):
            filepath = os.path.join(models_dir, filename)
            stat = os.stat(filepath)
            
            model_info = {
                "name": filename.replace('.pth', ''),
                "path": filepath,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
            }
            
            # 尝试读取元数据
            try:
                checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
                if 'config' in checkpoint:
                    model_info['base_model'] = checkpoint['config']._name_or_path
                if 'num_labels' in checkpoint:
                    model_info['num_labels'] = checkpoint['num_labels']
            except:
                pass
            
            models.append(model_info)
    
    return models


def predict_text(
    model, 
    tokenizer, 
    text: str, 
    device: str = 'cpu',
    max_length: int = 512
) -> Tuple[int, float, List[float]]:
    """
    对单条文本进行预测
    
    返回: (预测类别, 置信度, 各类别概率)
    """
    model.eval()
    
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    return prediction, confidence, probabilities[0].tolist()


def predict_batch(
    model, 
    tokenizer, 
    texts: List[str], 
    device: str = 'cpu',
    max_length: int = 512,
    batch_size: int = 16
) -> List[Tuple[int, float, List[float]]]:
    """批量预测"""
    results = []
    model.eval()
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            for j in range(len(batch_texts)):
                pred = predictions[j].item()
                conf = probabilities[j][pred].item()
                probs = probabilities[j].tolist()
                results.append((pred, conf, probs))
    
    return results


def interactive_mode(model, tokenizer, device: str = 'cpu', label_names: List[str] = None):
    """交互式预测模式"""
    print("\n" + "=" * 50)
    print("🤖 交互式预测模式")
    print("=" * 50)
    print("输入文本进行预测，输入 'quit' 或 'exit' 退出\n")
    
    while True:
        try:
            text = input("📝 请输入文本: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\n👋 再见！")
                break
            
            if not text:
                print("⚠️ 请输入有效文本\n")
                continue
            
            prediction, confidence, probabilities = predict_text(model, tokenizer, text, device)
            
            # 显示结果
            print("\n" + "-" * 40)
            if label_names and prediction < len(label_names):
                print(f"📊 预测类别: {prediction} ({label_names[prediction]})")
            else:
                print(f"📊 预测类别: {prediction}")
            print(f"✅ 置信度: {confidence:.2%}")
            print(f"📈 各类别概率: {[f'{p:.4f}' for p in probabilities]}")
            print("-" * 40 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 预测出错: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="模型预测命令行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='模型文件路径，例如: models/my_model.pth'
    )
    
    parser.add_argument(
        '--base-model', '-b',
        type=str,
        default='bert-base-uncased',
        help='基础模型名称（用于加载分词器），默认: bert-base-uncased'
    )
    
    parser.add_argument(
        '--text', '-t',
        type=str,
        help='要预测的文本'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='输入文件路径（每行一条文本）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='输出文件路径（CSV 格式）'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='启动交互式预测模式'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='列出所有可用的模型'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='计算设备，默认: auto（自动选择）'
    )
    
    parser.add_argument(
        '--labels',
        type=str,
        nargs='+',
        help='类别标签名称，例如: --labels 负面 正面'
    )
    
    args = parser.parse_args()
    
    # 列出可用模型
    if args.list:
        models = list_available_models()
        if not models:
            print("⚠️ models/ 目录下没有找到任何模型文件")
            print("💡 提示: 请先完成一次微调任务以生成模型")
        else:
            print("\n📦 可用模型列表:")
            print("-" * 60)
            for m in models:
                print(f"  • {m['name']}")
                print(f"    路径: {m['path']}")
                print(f"    大小: {m['size_mb']} MB")
                if 'base_model' in m:
                    print(f"    基础模型: {m['base_model']}")
                if 'num_labels' in m:
                    print(f"    类别数: {m['num_labels']}")
                print()
        return
    
    # 检查必要参数
    if not args.model:
        parser.error("请指定模型文件路径: --model models/xxx.pth")
    
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        print("💡 使用 --list 查看可用模型")
        sys.exit(1)
    
    # 确定设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"\n🔧 加载模型: {args.model}")
    print(f"💻 计算设备: {device.upper()}")
    
    # 加载模型和分词器
    try:
        # 先读取模型元数据以获取 base_model
        checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
        if 'config' in checkpoint and hasattr(checkpoint['config'], '_name_or_path'):
            base_model = checkpoint['config']._name_or_path
            print(f"📌 检测到基础模型: {base_model}")
        else:
            base_model = args.base_model
        
        tokenizer = load_tokenizer(base_model)
        model = load_saved_model(args.model, device=device)
        print("✅ 模型加载成功\n")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        sys.exit(1)
    
    # 交互式模式
    if args.interactive:
        interactive_mode(model, tokenizer, device, args.labels)
        return
    
    # 单条文本预测
    if args.text:
        prediction, confidence, probabilities = predict_text(model, tokenizer, args.text, device)
        
        print("=" * 50)
        print(f"📝 输入文本: {args.text}")
        print("-" * 50)
        
        if args.labels and prediction < len(args.labels):
            print(f"📊 预测类别: {prediction} ({args.labels[prediction]})")
        else:
            print(f"📊 预测类别: {prediction}")
        
        print(f"✅ 置信度: {confidence:.2%}")
        print(f"📈 各类别概率: {[f'{p:.4f}' for p in probabilities]}")
        print("=" * 50)
        return
    
    # 批量预测（从文件）
    if args.input:
        if not os.path.exists(args.input):
            print(f"❌ 输入文件不存在: {args.input}")
            sys.exit(1)
        
        # 读取输入文件
        with open(args.input, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"📄 读取 {len(texts)} 条文本")
        print("🔄 开始批量预测...")
        
        results = predict_batch(model, tokenizer, texts, device)
        
        # 输出结果
        output_file = args.output or args.input.replace('.txt', '_results.csv')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("text,prediction,confidence,probabilities\n")
            for text, (pred, conf, probs) in zip(texts, results):
                # 转义文本中的逗号和引号
                text_escaped = text.replace('"', '""')
                probs_str = ';'.join([f'{p:.4f}' for p in probs])
                f.write(f'"{text_escaped}",{pred},{conf:.4f},"{probs_str}"\n')
        
        print(f"✅ 预测完成！结果已保存到: {output_file}")
        
        # 显示统计信息
        predictions = [r[0] for r in results]
        for label in set(predictions):
            count = predictions.count(label)
            pct = count / len(predictions) * 100
            label_name = f" ({args.labels[label]})" if args.labels and label < len(args.labels) else ""
            print(f"   类别 {label}{label_name}: {count} 条 ({pct:.1f}%)")
        
        return
    
    # 如果没有指定任何操作，显示帮助
    parser.print_help()


if __name__ == "__main__":
    main()
