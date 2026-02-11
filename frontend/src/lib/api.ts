/**
 * API 配置文件
 * 统一管理 API 基础地址和请求配置
 */

// 从环境变量获取 API 地址，默认使用 localhost:8000
export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// API 请求超时时间（毫秒）
export const API_TIMEOUT = 60000;

// 通用的 fetch 封装，带错误处理
export async function apiFetch<T = any>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  
  const defaultHeaders: HeadersInit = {
    "Content-Type": "application/json",
  };

  try {
    const response = await fetch(url, {
      ...options,
      headers: {
        ...defaultHeaders,
        ...options.headers,
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new APIError(
        errorData.detail || `HTTP ${response.status}: ${response.statusText}`,
        response.status
      );
    }

    return response.json();
  } catch (error) {
    if (error instanceof APIError) {
      throw error;
    }
    throw new APIError(
      error instanceof Error ? error.message : "网络连接失败，请检查后端服务是否运行",
      0
    );
  }
}

// 自定义 API 错误类
export class APIError extends Error {
  status: number;
  
  constructor(message: string, status: number) {
    super(message);
    this.name = "APIError";
    this.status = status;
  }
}

// API 接口定义
export const api = {
  // GPU 状态
  gpu: {
    getStatus: () => apiFetch("/api/gpu/status"),
  },
  
  // 模型相关
  models: {
    // Ollama 模型列表
    list: () => apiFetch("/api/models"),
    // 数据库中的微调模型记录
    listFinetuned: () => apiFetch("/api/models/finetuned"),
    // 获取本地 models/ 目录下的所有模型文件
    listLocal: () => apiFetch<LocalModel[]>("/api/models/local"),
    // 获取单个本地模型的详细信息
    getLocalModel: (modelName: string) => apiFetch<LocalModelDetail>(`/api/models/local/${modelName}`),
    // 删除本地模型
    deleteLocal: (modelName: string) =>
      apiFetch(`/api/models/local/${modelName}`, { method: "DELETE" }),
    // 获取可用的基础模型列表（用于微调）
    listAvailableBase: () => apiFetch<BaseModelOption[]>("/api/models/available-base"),
    // 单条预测
    predict: (data: { model_path: string; text: string; base_model: string }) =>
      apiFetch<PredictResponse>("/api/models/predict", {
        method: "POST",
        body: JSON.stringify(data),
      }),
    // 批量预测
    batchPredict: (data: { texts: string[]; model_path: string; base_model: string }) =>
      apiFetch<BatchPredictResponse>("/api/models/predict/batch", {
        method: "POST",
        body: JSON.stringify(data),
      }),
  },
  
  // 微调相关
  finetune: {
    list: () => apiFetch("/api/finetune"),
    get: (taskId: string) => apiFetch(`/api/finetune/${taskId}`),
    create: (data: any) =>
      apiFetch("/api/finetune", {
        method: "POST",
        body: JSON.stringify(data),
      }),
    delete: (taskId: string) =>
      apiFetch(`/api/finetune/${taskId}`, { method: "DELETE" }),
    cancel: (taskId: string) =>
      apiFetch(`/api/finetune/${taskId}/cancel`, { method: "POST" }),
  },
  
  // Agent 相关
  agents: {
    list: () => apiFetch("/api/agents"),
    get: (agentId: string) => apiFetch(`/api/agents/${agentId}`),
    create: (data: any) =>
      apiFetch("/api/agents", {
        method: "POST",
        body: JSON.stringify(data),
      }),
    update: (agentId: string, data: any) =>
      apiFetch(`/api/agents/${agentId}`, {
        method: "PUT",
        body: JSON.stringify(data),
      }),
    delete: (agentId: string) =>
      apiFetch(`/api/agents/${agentId}`, { method: "DELETE" }),
  },
  
  // 聊天相关
  chat: {
    send: (data: any) =>
      apiFetch("/api/chat", {
        method: "POST",
        body: JSON.stringify(data),
      }),
    getHistory: (sessionId: string) => apiFetch(`/api/chat/history/${sessionId}`),
    getSessions: () => apiFetch("/api/chat/sessions"),
    deleteHistory: (sessionId: string) =>
      apiFetch(`/api/chat/history/${sessionId}`, { method: "DELETE" }),
  },
};

// ==================== 类型定义 ====================

// 本地模型信息
export interface LocalModel {
  name: string;
  filename: string;
  path: string;
  size_mb: number;
  created_at: string;
  modified_at: string;
  base_model?: string;
  num_labels?: number;
}

// 本地模型详细信息
export interface LocalModelDetail extends LocalModel {
  has_state_dict?: boolean;
}

// 基础模型选项
export interface BaseModelOption {
  name: string;
  description: string;
  parameters: string;
  recommended: boolean;
}

// 预测响应
export interface PredictResponse {
  text: string;
  prediction: number;
  confidence: number;
  probabilities: number[];
}

// 批量预测响应
export interface BatchPredictResponse {
  model_path: string;
  total: number;
  results: PredictResponse[];
}
