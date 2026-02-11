"use client";

import { Sidebar } from "@/components/Sidebar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Hammer, Play, CheckCircle, XCircle, Clock, RefreshCw, TestTube, Trash2, Cpu, Zap, Upload, StopCircle, Terminal, X } from "lucide-react";
import { useEffect, useState, useCallback, useRef } from "react";
import { useAppConfig } from "@/lib/config-context";
import { FileUpload } from "@/components/FileUpload";

// 预训练模型列表
const PRETRAINED_MODELS = [
  {
    value: "bert-base-uncased",
    label: "BERT Base (英文)",
    description: "英文基础BERT模型，适合英文文本分类",
    language: "英文",
  },
  {
    value: "bert-base-cased",
    label: "BERT Base Cased (英文)",
    description: "英文BERT模型，区分大小写",
    language: "英文",
  },
  {
    value: "bert-base-chinese",
    label: "BERT Base (中文)",
    description: "中文基础BERT模型，适合中文文本分类",
    language: "中文",
  },
  {
    value: "bert-base-multilingual-cased",
    label: "BERT Multilingual (多语言)",
    description: "支持104种语言的多语言BERT",
    language: "多语言",
  },
  {
    value: "hfl/chinese-bert-wwm-ext",
    label: "Chinese BERT WWM (中文增强)",
    description: "哈工大中文BERT，全词遮蔽，效果更好",
    language: "中文",
  },
  {
    value: "hfl/chinese-roberta-wwm-ext",
    label: "Chinese RoBERTa WWM (中文)",
    description: "哈工大中文RoBERTa，性能更强",
    language: "中文",
  },
  {
    value: "distilbert-base-uncased",
    label: "DistilBERT (英文轻量)",
    description: "BERT的轻量版本，速度快60%",
    language: "英文",
  },
  {
    value: "roberta-base",
    label: "RoBERTa Base (英文)",
    description: "优化版BERT，性能更强",
    language: "英文",
  },
];

interface FinetuneTask {
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
  status: string;
  progress: number;
  error_message?: string;
  model_path?: string;
  training_history?: any;
  metrics?: any;
  created_at: string;
  started_at?: string;
  completed_at?: string;
}

interface TestResult {
  text: string;
  prediction: number;
  confidence: number;
}

interface GpuStatus {
  cuda_available: boolean;
  cuda_version: string | null;
  device_count: number;
  devices: Array<{
    index: number;
    name: string;
    total_memory_gb: number;
    major: number;
    minor: number;
  }>;
  pytorch_version: string;
}

export default function FinetunePage() {
  const { config, getApiUrl } = useAppConfig();
  
  const [tasks, setTasks] = useState<FinetuneTask[]>([]);
  const [showForm, setShowForm] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [testingTaskId, setTestingTaskId] = useState<string | null>(null);
  const [testInput, setTestInput] = useState("");
  const [testResult, setTestResult] = useState<TestResult | null>(null);
  const [gpuStatus, setGpuStatus] = useState<GpuStatus | null>(null);
  const [showUpload, setShowUpload] = useState(false);
  const [logsTaskId, setLogsTaskId] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);
  
  const [formData, setFormData] = useState({
    base_model: config.defaultBaseModel || "bert-base-uncased",
    dataset_path: "",
    new_model_name: "",
    epochs: config.defaultEpochs || 3,
    learning_rate: config.defaultLearningRate || 2e-5,
    batch_size: config.defaultBatchSize || 8,
    max_length: config.defaultMaxLength || 128,
    text_column: "text",
    label_column: "target",
    use_gpu: config.useGpuByDefault ?? true,
    gradient_accumulation_steps: 4,
  });

  // 获取 GPU 状态
  const fetchGpuStatus = useCallback(async () => {
    try {
      const res = await fetch(getApiUrl("/api/gpu/status"));
      if (res.ok) {
        const data = await res.json();
        setGpuStatus(data);
      }
    } catch (error) {
      console.error("获取GPU状态失败:", error);
    }
  }, [getApiUrl]);

  // 加载任务列表
  const fetchTasks = useCallback(async () => {
    try {
      const res = await fetch(getApiUrl("/api/finetune"));
      if (res.ok) {
        const data = await res.json();
        setTasks(data);
        return data;
      }
    } catch (error) {
      console.error("获取任务列表失败:", error);
    }
    return [];
  }, [getApiUrl]);

  // 初始化加载任务和GPU状态
  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      await Promise.all([fetchTasks(), fetchGpuStatus()]);
      setIsLoading(false);
    };
    loadData();
  }, [fetchTasks, fetchGpuStatus]);

  // 轮询更新运行中的任务状态
  useEffect(() => {
    const hasRunningTask = tasks.some(
      (task) => task.status === "running" || task.status === "pending"
    );

    if (hasRunningTask) {
      // 如果有运行中的任务，每3秒轮询一次
      pollingRef.current = setInterval(() => {
        fetchTasks();
      }, 3000);
    } else {
      // 没有运行中的任务，停止轮询
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    }

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
      }
    };
  }, [tasks, fetchTasks]);

  // 提交微调任务
  const handleSubmit = async () => {
    if (!formData.dataset_path || !formData.new_model_name) {
      alert("请填写数据集路径和模型名称");
      return;
    }

    setIsSubmitting(true);
    try {
      const res = await fetch(getApiUrl("/api/finetune"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (res.ok) {
        const data = await res.json();
        alert(`微调任务已启动！任务ID: ${data.task_id}`);
        setShowForm(false);
        // 重新加载任务列表
        await fetchTasks();
        // 自动连接 WebSocket 查看日志
        connectWebSocket(data.task_id);
      } else {
        const error = await res.json();
        alert(`启动失败: ${error.detail || "未知错误"}`);
      }
    } catch (error) {
      console.error("提交微调任务失败:", error);
      alert("提交失败，请检查后端服务是否运行");
    } finally {
      setIsSubmitting(false);
    }
  };

  // 取消任务
  const handleCancelTask = async (taskId: string) => {
    if (!confirm("确定要取消这个任务吗？")) return;
    
    try {
      const res = await fetch(getApiUrl(`/api/finetune/${taskId}/cancel`), {
        method: "POST",
      });
      if (res.ok) {
        const data = await res.json();
        alert(data.message || "任务正在取消...");
        await fetchTasks();
      } else {
        const error = await res.json();
        alert(`取消失败: ${error.detail || "未知错误"}`);
      }
    } catch (error) {
      console.error("取消任务失败:", error);
    }
  };

  // 删除任务
  const handleDeleteTask = async (taskId: string) => {
    if (!confirm("确定要删除这个任务吗？")) return;
    
    try {
      const res = await fetch(getApiUrl(`/api/finetune/${taskId}`), {
        method: "DELETE",
      });
      if (res.ok) {
        setTasks((prev) => prev.filter((t) => t.id !== taskId));
      } else {
        alert("删除失败");
      }
    } catch (error) {
      console.error("删除任务失败:", error);
    }
  };

  // WebSocket 连接训练日志
  const connectWebSocket = (taskId: string) => {
    // 关闭旧连接
    if (wsRef.current) {
      wsRef.current.close();
    }
    
    setLogsTaskId(taskId);
    setLogs([]);
    
    // 构建 WebSocket URL
    const backendUrl = config.backendUrl || "http://localhost:8000";
    const wsUrl = backendUrl.replace(/^http/, "ws");
    const fullWsUrl = `${wsUrl}/ws/finetune/${taskId}`;
    
    setLogs((prev) => [...prev, `[系统] 正在连接: ${fullWsUrl}`]);
    
    try {
      const ws = new WebSocket(fullWsUrl);
      
      ws.onopen = () => {
        setLogs((prev) => [...prev, `[连接已建立] 正在监听任务 ${taskId} 的训练日志...`]);
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "log") {
            const timestamp = new Date(data.timestamp).toLocaleTimeString();
            setLogs((prev) => [...prev, `[${timestamp}] [${data.level.toUpperCase()}] ${data.message}`]);
          } else if (data.type === "progress") {
            setLogs((prev) => [...prev, `[进度] Epoch ${data.epoch}/${data.total_epochs} - ${data.progress.toFixed(1)}%`]);
          } else if (data.type === "connected") {
            setLogs((prev) => [...prev, `[系统] ${data.message}`]);
          } else if (data.type === "heartbeat") {
            // 忽略心跳消息
          }
        } catch {
          // 普通文本消息
          if (event.data !== "pong") {
            setLogs((prev) => [...prev, event.data]);
          }
        }
      };
      
      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        setLogs((prev) => [
          ...prev, 
          `[错误] WebSocket 连接失败`,
          `[提示] 请检查后端服务是否运行在 ${backendUrl}`,
          `[提示] 确保后端已启动并监听 WebSocket 连接`
        ]);
      };
      
      ws.onclose = (event) => {
        if (event.wasClean) {
          setLogs((prev) => [...prev, `[连接已关闭] code=${event.code}`]);
        } else {
          setLogs((prev) => [...prev, `[连接异常断开] code=${event.code}, reason=${event.reason || "未知"}`]);
        }
      };
      
      wsRef.current = ws;
    } catch (error) {
      setLogs((prev) => [...prev, `[错误] 创建 WebSocket 失败: ${error}`]);
    }
  };

  // 关闭 WebSocket 和日志面板
  const closeLogsPanel = () => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setLogsTaskId(null);
    setLogs([]);
  };

  // 自动滚动日志
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // 组件卸载时清理 WebSocket
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // 测试模型
  const handleTestModel = async (task: FinetuneTask) => {
    if (!testInput.trim()) {
      alert("请输入测试文本");
      return;
    }

    try {
      const res = await fetch(getApiUrl("/api/models/predict"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_path: task.model_path,
          text: testInput,
          base_model: task.base_model,
        }),
      });

      if (res.ok) {
        const result = await res.json();
        setTestResult({
          text: testInput,
          prediction: result.prediction,
          confidence: result.confidence,
        });
      } else {
        alert("预测失败，请检查模型是否可用");
      }
    } catch (error) {
      console.error("模型预测失败:", error);
      alert("模型预测失败");
    }
  };

  const handleInputChange = (field: string, value: string | number) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      case "failed":
        return <XCircle className="h-5 w-5 text-red-500" />;
      case "running":
        return <RefreshCw className="h-5 w-5 text-blue-500 animate-spin" />;
      default:
        return <Clock className="h-5 w-5 text-yellow-500" />;
    }
  };

  const getStatusText = (status: string) => {
    const statusMap: Record<string, string> = {
      pending: "等待中",
      running: "运行中",
      completed: "已完成",
      failed: "失败",
    };
    return statusMap[status] || status;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed":
        return "text-green-500";
      case "failed":
        return "text-red-500";
      case "running":
        return "text-blue-500";
      default:
        return "text-yellow-500";
    }
  };

  // 格式化时间
  const formatTime = (isoString?: string) => {
    if (!isoString) return "-";
    const date = new Date(isoString);
    return date.toLocaleString("zh-CN");
  };

  // 计算耗时
  const calcDuration = (task: FinetuneTask) => {
    if (!task.started_at) return null;
    const start = new Date(task.started_at).getTime();
    const end = task.completed_at
      ? new Date(task.completed_at).getTime()
      : Date.now();
    const seconds = Math.round((end - start) / 1000);
    if (seconds < 60) return `${seconds}秒`;
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes}分${secs}秒`;
  };

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 bg-muted/10 p-8">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">模型微调</h1>
            <p className="text-muted-foreground">
              使用您的数据集微调预训练模型
            </p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={fetchTasks}
              className="flex items-center gap-2 rounded-lg border px-4 py-2 hover:bg-accent"
              aria-label="刷新任务列表"
              tabIndex={0}
            >
              <RefreshCw className="h-4 w-4" />
              刷新
            </button>
            <button
              onClick={() => setShowForm(!showForm)}
              className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-primary-foreground hover:bg-primary/90"
              aria-label="新建微调任务"
              tabIndex={0}
            >
              <Hammer className="h-4 w-4" />
              新建微调任务
            </button>
          </div>
        </div>

        {/* GPU 状态显示 */}
        <Card className="mb-6">
          <CardContent className="py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {gpuStatus?.cuda_available ? (
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-green-100">
                    <Zap className="h-5 w-5 text-green-600" />
                  </div>
                ) : (
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gray-100">
                    <Cpu className="h-5 w-5 text-gray-600" />
                  </div>
                )}
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-medium">
                      {gpuStatus?.cuda_available ? "🚀 GPU 加速可用" : "💻 仅 CPU 模式"}
                    </span>
                    {gpuStatus?.cuda_available && (
                      <span className="rounded-full bg-green-100 px-2 py-0.5 text-xs font-medium text-green-700">
                        CUDA {gpuStatus.cuda_version}
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {gpuStatus?.cuda_available && gpuStatus.devices.length > 0
                      ? `${gpuStatus.devices[0].name} (${gpuStatus.devices[0].total_memory_gb} GB)`
                      : gpuStatus?.cuda_available === false
                      ? "PyTorch GPU 版本未安装，建议安装以加速训练"
                      : "正在检测..."
                    }
                  </p>
                </div>
              </div>
              <div className="text-right text-sm text-muted-foreground">
                <p>PyTorch: {gpuStatus?.pytorch_version || "-"}</p>
                {!gpuStatus?.cuda_available && (
                  <p className="text-xs text-amber-600 mt-1">
                    运行: pip install torch --index-url https://download.pytorch.org/whl/cu121
                  </p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* 微调表单 */}
        {showForm && (
          <Card className="mb-6">
            <CardHeader>
              <CardTitle>配置微调任务</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="mb-2 block text-sm font-medium">基础模型 *</label>
                  <select
                    value={formData.base_model}
                    onChange={(e) => handleInputChange("base_model", e.target.value)}
                    className="w-full rounded-lg border bg-background px-3 py-2 cursor-pointer"
                    aria-label="选择基础模型"
                  >
                    {PRETRAINED_MODELS.map((model) => (
                      <option key={model.value} value={model.value}>
                        {model.label}
                      </option>
                    ))}
                  </select>
                  {/* 显示选中模型的描述 */}
                  <p className="mt-1 text-xs text-muted-foreground">
                    {PRETRAINED_MODELS.find((m) => m.value === formData.base_model)?.description || ""}
                  </p>
                </div>
                <div>
                  <label className="mb-2 block text-sm font-medium">新模型名称 *</label>
                  <input
                    type="text"
                    value={formData.new_model_name}
                    onChange={(e) => handleInputChange("new_model_name", e.target.value)}
                    className="w-full rounded-lg border bg-background px-3 py-2"
                    placeholder="my-custom-model"
                  />
                </div>
              </div>

              <div>
                <div className="mb-2 flex items-center justify-between">
                  <label className="text-sm font-medium">数据集路径 *</label>
                  <button
                    type="button"
                    onClick={() => setShowUpload(!showUpload)}
                    className="flex items-center gap-1 text-xs text-primary hover:underline"
                  >
                    <Upload className="h-3 w-3" />
                    {showUpload ? "手动输入" : "上传文件"}
                  </button>
                </div>
                
                {showUpload ? (
                  <FileUpload
                    onUploadSuccess={(filePath) => {
                      handleInputChange("dataset_path", filePath);
                      setShowUpload(false);
                    }}
                    accept=".csv,.json"
                    maxSize={50}
                  />
                ) : (
                  <>
                    <input
                      type="text"
                      value={formData.dataset_path}
                      onChange={(e) => handleInputChange("dataset_path", e.target.value)}
                      className="w-full rounded-lg border bg-background px-3 py-2"
                      placeholder="data/sample_train.csv"
                    />
                    <p className="mt-1 text-xs text-muted-foreground">支持 CSV 或 JSON 格式，示例：data/sample_train.csv</p>
                  </>
                )}
              </div>

              <div className="grid gap-4 md:grid-cols-4">
                <div>
                  <label className="mb-2 block text-sm font-medium">训练轮数</label>
                  <input
                    type="number"
                    value={formData.epochs}
                    onChange={(e) => handleInputChange("epochs", parseInt(e.target.value))}
                    className="w-full rounded-lg border bg-background px-3 py-2"
                    min={1}
                    max={100}
                  />
                </div>
                <div>
                  <label className="mb-2 block text-sm font-medium">学习率</label>
                  <input
                    type="text"
                    value={formData.learning_rate}
                    onChange={(e) => handleInputChange("learning_rate", parseFloat(e.target.value))}
                    className="w-full rounded-lg border bg-background px-3 py-2"
                  />
                </div>
                <div>
                  <label className="mb-2 block text-sm font-medium">批次大小</label>
                  <input
                    type="number"
                    value={formData.batch_size}
                    onChange={(e) => handleInputChange("batch_size", parseInt(e.target.value))}
                    className="w-full rounded-lg border bg-background px-3 py-2"
                    min={1}
                    max={64}
                  />
                  <p className="mt-1 text-xs text-muted-foreground">
                    推荐: 4-16 (显存不足时减小)
                  </p>
                </div>
                <div>
                  <label className="mb-2 block text-sm font-medium">最大长度</label>
                  <input
                    type="number"
                    value={formData.max_length}
                    onChange={(e) => handleInputChange("max_length", parseInt(e.target.value))}
                    className="w-full rounded-lg border bg-background px-3 py-2"
                    min={32}
                    max={512}
                  />
                  <p className="mt-1 text-xs text-muted-foreground">
                    推荐: 64-256 (显存不足时减小)
                  </p>
                </div>
                <div>
                  <label className="mb-2 block text-sm font-medium">梯度累积步数</label>
                  <input
                    type="number"
                    value={formData.gradient_accumulation_steps}
                    onChange={(e) => handleInputChange("gradient_accumulation_steps", parseInt(e.target.value))}
                    className="w-full rounded-lg border bg-background px-3 py-2"
                    min={1}
                    max={32}
                  />
                  <p className="mt-1 text-xs text-muted-foreground">
                    等效batch_size: {formData.batch_size * formData.gradient_accumulation_steps}
                  </p>
                </div>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="mb-2 block text-sm font-medium">文本列名</label>
                  <input
                    type="text"
                    value={formData.text_column}
                    onChange={(e) => handleInputChange("text_column", e.target.value)}
                    className="w-full rounded-lg border bg-background px-3 py-2"
                    placeholder="text"
                  />
                </div>
                <div>
                  <label className="mb-2 block text-sm font-medium">标签列名</label>
                  <input
                    type="text"
                    value={formData.label_column}
                    onChange={(e) => handleInputChange("label_column", e.target.value)}
                    className="w-full rounded-lg border bg-background px-3 py-2"
                    placeholder="target"
                  />
                </div>
              </div>

              {/* GPU 加速开关 */}
              <div className="rounded-lg border bg-muted/30 p-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {gpuStatus?.cuda_available ? (
                      <Zap className="h-5 w-5 text-green-500" />
                    ) : (
                      <Cpu className="h-5 w-5 text-gray-400" />
                    )}
                    <div>
                      <label className="font-medium">GPU 加速</label>
                      <p className="text-sm text-muted-foreground">
                        {gpuStatus?.cuda_available
                          ? `使用 ${gpuStatus.devices[0]?.name || "GPU"} 加速训练`
                          : "GPU 不可用，将使用 CPU 训练（速度较慢）"
                        }
                      </p>
                    </div>
                  </div>
                  <label className="relative inline-flex cursor-pointer items-center">
                    <input
                      type="checkbox"
                      checked={formData.use_gpu}
                      onChange={(e) => handleInputChange("use_gpu", e.target.checked ? true : false)}
                      disabled={!gpuStatus?.cuda_available}
                      className="peer sr-only"
                    />
                    <div className={`h-6 w-11 rounded-full transition-colors after:absolute after:left-[2px] after:top-[2px] after:h-5 after:w-5 after:rounded-full after:border after:border-gray-300 after:bg-white after:transition-all after:content-[''] peer-checked:bg-green-500 peer-checked:after:translate-x-full peer-checked:after:border-white peer-disabled:cursor-not-allowed peer-disabled:opacity-50 ${gpuStatus?.cuda_available ? 'bg-gray-200' : 'bg-gray-100'}`}></div>
                  </label>
                </div>
                {!gpuStatus?.cuda_available && (
                  <p className="mt-2 text-xs text-amber-600">
                    💡 安装 GPU 版 PyTorch 可大幅提升训练速度: pip install torch --index-url https://download.pytorch.org/whl/cu121
                  </p>
                )}
              </div>

              <div className="flex gap-2">
                <button
                  onClick={handleSubmit}
                  disabled={isSubmitting}
                  className="rounded-lg bg-primary px-4 py-2 text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                >
                  {isSubmitting ? "提交中..." : "开始微调"}
                </button>
                <button
                  onClick={() => setShowForm(false)}
                  className="rounded-lg border px-4 py-2 hover:bg-accent"
                >
                  取消
                </button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* 任务列表 */}
        {isLoading ? (
          <Card>
            <CardContent className="flex items-center justify-center py-12">
              <RefreshCw className="mr-2 h-6 w-6 animate-spin" />
              <p>加载中...</p>
            </CardContent>
          </Card>
        ) : tasks.length === 0 ? (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <Hammer className="mb-4 h-12 w-12 text-muted-foreground" />
              <p className="text-lg font-medium">暂无微调任务</p>
              <p className="text-muted-foreground">点击上方按钮创建您的第一个微调任务</p>
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-4">
            {tasks.map((task) => (
              <Card key={task.id} className="overflow-hidden">
                <CardContent className="p-0">
                  {/* 主要信息行 */}
                  <div className="flex items-center justify-between p-4">
                    <div className="flex items-center gap-4">
                      {getStatusIcon(task.status)}
                      <div>
                        <p className="font-medium text-lg">{task.new_model_name}</p>
                        <p className="text-sm text-muted-foreground">
                          基于 {task.base_model} 
                          <span className="ml-1 text-xs px-1.5 py-0.5 rounded bg-muted">
                            {PRETRAINED_MODELS.find((m) => m.value === task.base_model)?.language || "未知"}
                          </span>
                          {" "}| {task.epochs} 轮训练
                        </p>
                        <p className="text-xs text-muted-foreground mt-1">
                          创建于: {formatTime(task.created_at)}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <p className={`font-medium ${getStatusColor(task.status)}`}>
                          {getStatusText(task.status)}
                        </p>
                        {calcDuration(task) && (
                          <p className="text-sm text-muted-foreground">
                            耗时: {calcDuration(task)}
                          </p>
                        )}
                      </div>
                      <div className="flex gap-2">
                        {/* 运行中的任务显示取消和日志按钮 */}
                        {(task.status === "running" || task.status === "pending") && (
                          <>
                            <button
                              onClick={() => connectWebSocket(task.id)}
                              className="flex items-center gap-1 rounded-lg border px-3 py-1.5 text-sm hover:bg-accent"
                              aria-label="查看日志"
                              tabIndex={0}
                            >
                              <Terminal className="h-4 w-4" />
                              日志
                            </button>
                            <button
                              onClick={() => handleCancelTask(task.id)}
                              className="flex items-center gap-1 rounded-lg border border-orange-200 px-3 py-1.5 text-sm text-orange-500 hover:bg-orange-50"
                              aria-label="取消任务"
                              tabIndex={0}
                            >
                              <StopCircle className="h-4 w-4" />
                              取消
                            </button>
                          </>
                        )}
                        {task.status === "completed" && (
                          <button
                            onClick={() => setTestingTaskId(testingTaskId === task.id ? null : task.id)}
                            className="flex items-center gap-1 rounded-lg border px-3 py-1.5 text-sm hover:bg-accent"
                            aria-label="测试模型"
                            tabIndex={0}
                          >
                            <TestTube className="h-4 w-4" />
                            测试
                          </button>
                        )}
                        <button
                          onClick={() => handleDeleteTask(task.id)}
                          className="flex items-center gap-1 rounded-lg border border-red-200 px-3 py-1.5 text-sm text-red-500 hover:bg-red-50"
                          aria-label="删除任务"
                          tabIndex={0}
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </div>
                  </div>

                  {/* 进度条 - 运行中时显示 */}
                  {(task.status === "running" || task.status === "pending") && (
                    <div className="px-4 pb-4">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm text-muted-foreground">训练进度</span>
                        <span className="text-sm font-medium">{task.progress || 0}%</span>
                      </div>
                      <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
                        <div
                          className="h-full bg-blue-500 transition-all duration-500"
                          style={{ width: `${task.progress || 0}%` }}
                        />
                      </div>
                    </div>
                  )}

                  {/* 错误信息 */}
                  {task.status === "failed" && task.error_message && (
                    <div className="mx-4 mb-4 rounded-lg bg-red-50 p-3 text-sm text-red-600">
                      <strong>错误信息：</strong> {task.error_message}
                    </div>
                  )}

                  {/* 完成信息 */}
                  {task.status === "completed" && (
                    <div className="mx-4 mb-4 rounded-lg bg-green-50 p-3 text-sm text-green-700">
                      <p><strong>模型路径：</strong> {task.model_path}</p>
                      {task.training_history && task.training_history.length > 0 && (
                        <p className="mt-1">
                          <strong>最终指标：</strong> 
                          Loss: {task.training_history[task.training_history.length - 1]?.val_loss?.toFixed(4) || 'N/A'} | 
                          Acc: {((task.training_history[task.training_history.length - 1]?.val_acc || 0) * 100).toFixed(1)}%
                        </p>
                      )}
                    </div>
                  )}

                  {/* 模型测试面板 */}
                  {testingTaskId === task.id && task.status === "completed" && (
                    <div className="border-t bg-muted/30 p-4">
                      <h4 className="mb-3 font-medium">模型测试</h4>
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={testInput}
                          onChange={(e) => setTestInput(e.target.value)}
                          placeholder="输入测试文本..."
                          className="flex-1 rounded-lg border bg-background px-3 py-2"
                        />
                        <button
                          onClick={() => handleTestModel(task)}
                          className="rounded-lg bg-primary px-4 py-2 text-primary-foreground hover:bg-primary/90"
                        >
                          预测
                        </button>
                      </div>
                      {testResult && (
                        <div className="mt-3 rounded-lg bg-background p-3 space-y-2">
                          <p><strong>输入：</strong> {testResult.text}</p>
                          <div className="flex items-center gap-2">
                            <strong>预测结果：</strong>
                            <span className={`inline-flex items-center gap-1 rounded-full px-3 py-1 text-sm font-medium ${
                              testResult.prediction === 1 
                                ? "bg-green-100 text-green-700" 
                                : "bg-red-100 text-red-700"
                            }`}>
                              {testResult.prediction === 1 ? "👍 正面" : "👎 负面"}
                              <span className="text-xs opacity-70">(标签: {testResult.prediction})</span>
                            </span>
                          </div>
                          <p><strong>置信度：</strong> {(testResult.confidence * 100).toFixed(2)}%</p>
                          {/* 语言提示 */}
                          {(() => {
                            const modelInfo = PRETRAINED_MODELS.find((m) => m.value === task.base_model);
                            const isChineseInput = /[\u4e00-\u9fa5]/.test(testResult.text);
                            const isChineseModel = modelInfo?.language?.includes("中文") || modelInfo?.language === "多语言";
                            if (isChineseInput && !isChineseModel) {
                              return (
                                <div className="mt-2 rounded bg-yellow-50 p-2 text-xs text-yellow-700">
                                  ⚠️ 提示：检测到输入为中文，但当前模型 ({modelInfo?.label}) 不支持中文。建议使用中文或多语言模型重新训练。
                                </div>
                              );
                            }
                            return null;
                          })()}
                        </div>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        {/* 训练日志面板 */}
        {logsTaskId && (
          <div className="fixed bottom-0 left-64 right-0 z-50 border-t bg-card shadow-lg">
            <div className="flex items-center justify-between border-b px-4 py-2">
              <div className="flex items-center gap-2">
                <Terminal className="h-4 w-4 text-green-500" />
                <span className="font-medium">训练日志</span>
                <span className="text-xs text-muted-foreground">
                  任务: {logsTaskId.slice(0, 8)}...
                </span>
              </div>
              <button
                onClick={closeLogsPanel}
                className="rounded p-1 hover:bg-muted"
                aria-label="关闭日志"
                tabIndex={0}
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="h-48 overflow-y-auto bg-gray-900 p-4 font-mono text-sm text-green-400">
              {logs.length === 0 ? (
                <p className="text-gray-500">等待日志...</p>
              ) : (
                logs.map((log, index) => (
                  <div key={index} className="whitespace-pre-wrap">
                    {log}
                  </div>
                ))
              )}
              <div ref={logsEndRef} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
