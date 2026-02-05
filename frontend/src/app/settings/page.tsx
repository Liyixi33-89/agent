"use client";

import { Sidebar } from "@/components/Sidebar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Settings, Server, Database, Globe } from "lucide-react";
import { useState } from "react";

export default function SettingsPage() {
  const [ollamaUrl, setOllamaUrl] = useState("http://localhost:11434");
  const [backendUrl, setBackendUrl] = useState("http://localhost:8000");

  const handleSave = () => {
    // 这里可以保存到 localStorage 或发送到后端
    localStorage.setItem("ollamaUrl", ollamaUrl);
    localStorage.setItem("backendUrl", backendUrl);
    alert("设置已保存");
  };

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 bg-muted/10 p-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold tracking-tight">系统设置</h1>
          <p className="text-muted-foreground">配置平台的各项参数</p>
        </div>

        <div className="max-w-2xl space-y-6">
          {/* API 配置 */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Server className="h-5 w-5" />
                服务配置
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="mb-2 block text-sm font-medium">后端服务地址</label>
                <input
                  type="text"
                  value={backendUrl}
                  onChange={(e) => setBackendUrl(e.target.value)}
                  className="w-full rounded-lg border bg-background px-3 py-2"
                  placeholder="http://localhost:8000"
                />
                <p className="mt-1 text-xs text-muted-foreground">
                  Python FastAPI 后端服务地址
                </p>
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium">Ollama 服务地址</label>
                <input
                  type="text"
                  value={ollamaUrl}
                  onChange={(e) => setOllamaUrl(e.target.value)}
                  className="w-full rounded-lg border bg-background px-3 py-2"
                  placeholder="http://localhost:11434"
                />
                <p className="mt-1 text-xs text-muted-foreground">
                  本地 Ollama 服务地址，用于模型推理
                </p>
              </div>
            </CardContent>
          </Card>

          {/* 模型配置 */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="h-5 w-5" />
                默认模型配置
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="mb-2 block text-sm font-medium">默认微调基础模型</label>
                <input
                  type="text"
                  defaultValue="bert-base-uncased"
                  className="w-full rounded-lg border bg-background px-3 py-2"
                  placeholder="bert-base-uncased"
                />
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium">模型保存路径</label>
                <input
                  type="text"
                  defaultValue="./models"
                  className="w-full rounded-lg border bg-background px-3 py-2"
                  placeholder="./models"
                />
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

          <button
            onClick={handleSave}
            className="rounded-lg bg-primary px-6 py-2 text-primary-foreground hover:bg-primary/90"
          >
            保存设置
          </button>
        </div>
      </main>
    </div>
  );
}
