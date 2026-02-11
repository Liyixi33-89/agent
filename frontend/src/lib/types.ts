/**
 * 共享类型定义
 */

// 微调任务状态
export type TaskStatus = "pending" | "running" | "completed" | "failed" | "cancelled";

// 微调任务
export interface FinetuneTask {
  id: string;
  base_model: string;
  new_model_name: string;
  dataset_path: string;
  epochs: number;
  learning_rate: number;
  batch_size: number;
  max_length: number;
  text_column: string;
  label_column: string;
  use_gpu: boolean;
  gradient_accumulation_steps?: number;
  status: TaskStatus;
  progress: number;
  error_message?: string;
  model_path?: string;
  training_history?: TrainingHistory;
  metrics?: Record<string, number>;
  created_at: string;
  started_at?: string;
  completed_at?: string;
}

// 训练历史
export interface TrainingHistory {
  train_loss: number[];
  train_accuracy: number[];
  val_loss: number[];
  val_accuracy: number[];
  val_precision?: number[];
  val_recall?: number[];
  val_f1?: number[];
}

// Agent 配置
export interface Agent {
  id?: string;
  name: string;
  role: string;
  system_prompt: string;
  model: string;
  is_active?: boolean;
  config?: Record<string, any>;
  created_at?: string;
  updated_at?: string;
}

// 聊天消息
export interface ChatMessage {
  id?: string;
  role: "user" | "assistant" | "system";
  content: string;
  model_used?: string;
  tokens_used?: number;
  created_at?: string;
}

// GPU 状态
export interface GpuStatus {
  cuda_available: boolean;
  cuda_version: string | null;
  device_count: number;
  devices: GpuDevice[];
  pytorch_version: string;
}

export interface GpuDevice {
  index: number;
  name: string;
  total_memory_gb: number;
  major: number;
  minor: number;
}

// 模型信息
export interface ModelInfo {
  id?: string;
  name: string;
  base_model?: string;
  model_type: "ollama" | "finetuned" | "huggingface";
  path?: string;
  status?: string;
  description?: string;
  parameters?: Record<string, any>;
  metrics?: Record<string, number>;
  finetune_task_id?: string;
  created_at?: string;
  updated_at?: string;
}

// 预训练模型选项
export interface PretrainedModelOption {
  value: string;
  label: string;
  description: string;
  language: string;
}

// 预测结果
export interface PredictResult {
  text: string;
  prediction: number;
  confidence: number;
  probabilities?: number[];
}
