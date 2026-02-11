"use client";

import { Sidebar } from "@/components/Sidebar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Settings, Server, Database, Globe, RefreshCw, Check, AlertCircle } from "lucide-react";
import { useState, useEffect } from "react";
import { useAppConfig, type AppConfig } from "@/lib/config-context";

export default function SettingsPage() {
  const { config, updateConfig, resetConfig } = useAppConfig();
  const [localConfig, setLocalConfig] = useState<Partial<AppConfig>>({});
  const [isSaving, setIsSaving] = useState(false);
  const [saveStatus, setSaveStatus] = useState<"idle" | "success" | "error">("idle");
  const [connectionStatus, setConnectionStatus] = useState<{
    backend: "checking" | "connected" | "error";
    ollama: "checking" | "connected" | "error";
  }>({
    backend: "checking",
    ollama: "checking",
  });

  // 初始化本地配置
  useEffect(() => {
    setLocalConfig({
      backendUrl: config.backendUrl,
      ollamaUrl: config.ollamaUrl,
      defaultBaseModel: config.defaultBaseModel,
      modelSavePath: config.modelSavePath,
      defaultBatchSize: config.defaultBatchSize,
      defaultMaxLength: config.defaultMaxLength,
      defaultEpochs: config.defaultEpochs,
      defaultLearningRate: config.defaultLearningRate,
      useGpuByDefault: config.useGpuByDefault,
    });
  }, [config]);

  // 检查服务连接状态
  const checkConnections = async () => {
    setConnectionStatus({ backend: "checking", ollama: "checking" });

    // 检查后端
    try {
      const backendUrl = localConfig.backendUrl || config.backendUrl;
      const res = await fetch(`${backendUrl}/`, { 
        method: "GET",
        signal: AbortSignal.timeout(5000)
      });
      setConnectionStatus((prev) => ({
        ...prev,
        backend: res.ok ? "connected" : "error",
      }));
    } catch {
      setConnectionStatus((prev) => ({ ...prev, backend: "error" }));
    }

    // 检查 Ollama（通过后端代理）
    try {
      const backendUrl = localConfig.backendUrl || config.backendUrl;
      const res = await fetch(`${backendUrl}/api/models`, {
        method: "GET",
        signal: AbortSignal.timeout(5000)
      });
      const data = await res.json();
      setConnectionStatus((prev) => ({
        ...prev,
        ollama: data.error ? "error" : "connected",
      }));
    } catch {
      setConnectionStatus((prev) => ({ ...prev, ollama: "error" }));
    }
  };

  // 初始检查连接
  useEffect(() => {
    checkConnections();
  }, []);

  const handleInputChange = (key: keyof AppConfig, value: string | number | boolean) => {
    setLocalConfig((prev) => ({ ...prev, [key]: value }));
    setSaveStatus("idle");
  };

  const handleSave = async () => {
    setIsSaving(true);
    setSaveStatus("idle");

    try {
      updateConfig(localConfig);
      setSaveStatus("success");
      // 保存后重新检查连接
      setTimeout(checkConnections, 500);
    } catch (error) {
      console.error("保存配置失败:", error);
      setSaveStatus("error");
    } finally {
      setIsSaving(false);
    }
  };

  const handleReset = () => {
    if (confirm("确定要重置所有设置为默认值吗？")) {
      resetConfig();
      setSaveStatus("idle");
      setTimeout(checkConnections, 500);
    }
  };

  const getStatusIcon = (status: "checking" | "connected" | "error") => {
    switch (status) {
      case "checking":
        return <RefreshCw className="h-4 w-4 animate-spin text-muted-foreground" />;
      case "connected":
        return <Check className="h-4 w-4 text-green-500" />;
      case "error":
        return <AlertCircle className="h-4 w-4 text-red-500" />;
    }
  };

  const getStatusText = (status: "checking" | "connected" | "error") => {
    switch (status) {
      case "checking":
        return "检查中...";
      case "connected":
        return "已连接";
      case "error":
        return "连接失败";
    }
  };

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 bg-muted/10 p-8">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">系统设置</h1>
            <p className="text-muted-foreground">配置平台的各项参数</p>
          </div>
          <button
            onClick={checkConnections}
            className="flex items-center gap-2 rounded-lg border px-4 py-2 hover:bg-accent"
            aria-label="检查连接状态"
            tabIndex={0}
          >
            <RefreshCw className="h-4 w-4" />
            检查连接
          </button>
        </div>

        <div className="max-w-2xl space-y-6">
          {/* 服务配置 */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Server className="h-5 w-5" />
                服务配置
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <div className="mb-2 flex items-center justify-between">
                  <label className="text-sm font-medium">后端服务地址</label>
                  <div className="flex items-center gap-1 text-sm">
                    {getStatusIcon(connectionStatus.backend)}
                    <span className={connectionStatus.backend === "connected" ? "text-green-600" : connectionStatus.backend === "error" ? "text-red-600" : "text-muted-foreground"}>
                      {getStatusText(connectionStatus.backend)}
                    </span>
                  </div>
                </div>
                <input
                  type="text"
                  value={localConfig.backendUrl || ""}
                  onChange={(e) => handleInputChange("backendUrl", e.target.value)}
                  className="w-full rounded-lg border bg-background px-3 py-2"
                  placeholder="http://localhost:8000"
                />
                <p className="mt-1 text-xs text-muted-foreground">
                  Python FastAPI 后端服务地址
                </p>
              </div>
              <div>
                <div className="mb-2 flex items-center justify-between">
                  <label className="text-sm font-medium">Ollama 服务地址</label>
                  <div className="flex items-center gap-1 text-sm">
                    {getStatusIcon(connectionStatus.ollama)}
                    <span className={connectionStatus.ollama === "connected" ? "text-green-600" : connectionStatus.ollama === "error" ? "text-red-600" : "text-muted-foreground"}>
                      {getStatusText(connectionStatus.ollama)}
                    </span>
                  </div>
                </div>
                <input
                  type="text"
                  value={localConfig.ollamaUrl || ""}
                  onChange={(e) => handleInputChange("ollamaUrl", e.target.value)}
                  className="w-full rounded-lg border bg-background px-3 py-2"
                  placeholder="http://localhost:11434"
                />
                <p className="mt-1 text-xs text-muted-foreground">
                  本地 Ollama 服务地址，用于模型推理
                </p>
              </div>
            </CardContent>
          </Card>

          {/* 微调默认配置 */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="h-5 w-5" />
                微调默认配置
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="mb-2 block text-sm font-medium">默认基础模型</label>
                  <input
                    type="text"
                    value={localConfig.defaultBaseModel || ""}
                    onChange={(e) => handleInputChange("defaultBaseModel", e.target.value)}
                    className="w-full rounded-lg border bg-background px-3 py-2"
                    placeholder="bert-base-uncased"
                  />
                </div>
                <div>
                  <label className="mb-2 block text-sm font-medium">模型保存路径</label>
                  <input
                    type="text"
                    value={localConfig.modelSavePath || ""}
                    onChange={(e) => handleInputChange("modelSavePath", e.target.value)}
                    className="w-full rounded-lg border bg-background px-3 py-2"
                    placeholder="./models"
                  />
                </div>
              </div>

              <div className="grid gap-4 md:grid-cols-3">
                <div>
                  <label className="mb-2 block text-sm font-medium">默认训练轮数</label>
                  <input
                    type="number"
                    value={localConfig.defaultEpochs || 3}
                    onChange={(e) => handleInputChange("defaultEpochs", parseInt(e.target.value))}
                    className="w-full rounded-lg border bg-background px-3 py-2"
                    min={1}
                    max={100}
                  />
                </div>
                <div>
                  <label className="mb-2 block text-sm font-medium">默认批次大小</label>
                  <input
                    type="number"
                    value={localConfig.defaultBatchSize || 8}
                    onChange={(e) => handleInputChange("defaultBatchSize", parseInt(e.target.value))}
                    className="w-full rounded-lg border bg-background px-3 py-2"
                    min={1}
                    max={64}
                  />
                </div>
                <div>
                  <label className="mb-2 block text-sm font-medium">默认最大长度</label>
                  <input
                    type="number"
                    value={localConfig.defaultMaxLength || 128}
                    onChange={(e) => handleInputChange("defaultMaxLength", parseInt(e.target.value))}
                    className="w-full rounded-lg border bg-background px-3 py-2"
                    min={32}
                    max={512}
                  />
                </div>
              </div>

              <div>
                <label className="mb-2 block text-sm font-medium">默认学习率</label>
                <input
                  type="text"
                  value={localConfig.defaultLearningRate || ""}
                  onChange={(e) => handleInputChange("defaultLearningRate", parseFloat(e.target.value) || 2e-5)}
                  className="w-full rounded-lg border bg-background px-3 py-2"
                  placeholder="2e-5"
                />
              </div>

              {/* GPU 开关 */}
              <div className="flex items-center justify-between rounded-lg border bg-muted/30 p-4">
                <div>
                  <label className="font-medium">默认启用 GPU 加速</label>
                  <p className="text-sm text-muted-foreground">
                    新建微调任务时默认启用 GPU（如果可用）
                  </p>
                </div>
                <label className="relative inline-flex cursor-pointer items-center">
                  <input
                    type="checkbox"
                    checked={localConfig.useGpuByDefault ?? true}
                    onChange={(e) => handleInputChange("useGpuByDefault", e.target.checked)}
                    className="peer sr-only"
                  />
                  <div className="h-6 w-11 rounded-full bg-gray-200 transition-colors after:absolute after:left-[2px] after:top-[2px] after:h-5 after:w-5 after:rounded-full after:border after:border-gray-300 after:bg-white after:transition-all after:content-[''] peer-checked:bg-green-500 peer-checked:after:translate-x-full peer-checked:after:border-white"></div>
                </label>
              </div>
            </CardContent>
          </Card>

          {/* 关于 */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Globe className="h-5 w-5" />
                关于
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-sm text-muted-foreground">
                <p><strong>Agent 微调平台</strong> v1.0.0</p>
                <p>一个用于管理和微调 AI 智能体的平台</p>
                <p>技术栈：Next.js + FastAPI + Transformers + Ollama</p>
              </div>
            </CardContent>
          </Card>

          {/* 操作按钮 */}
          <div className="flex items-center gap-4">
            <button
              onClick={handleSave}
              disabled={isSaving}
              className="flex items-center gap-2 rounded-lg bg-primary px-6 py-2 text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
              aria-label="保存设置"
              tabIndex={0}
            >
              {isSaving ? (
                <RefreshCw className="h-4 w-4 animate-spin" />
              ) : saveStatus === "success" ? (
                <Check className="h-4 w-4" />
              ) : (
                <Settings className="h-4 w-4" />
              )}
              {isSaving ? "保存中..." : saveStatus === "success" ? "已保存" : "保存设置"}
            </button>
            <button
              onClick={handleReset}
              className="rounded-lg border px-6 py-2 hover:bg-accent"
              aria-label="重置设置"
              tabIndex={0}
            >
              重置为默认
            </button>

            {saveStatus === "success" && (
              <span className="text-sm text-green-600">✓ 设置已保存到本地存储</span>
            )}
            {saveStatus === "error" && (
              <span className="text-sm text-red-600">✗ 保存失败，请重试</span>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
