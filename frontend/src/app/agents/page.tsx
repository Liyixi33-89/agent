"use client";

import { Sidebar } from "@/components/Sidebar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Bot, Plus, Trash2, Pencil, X } from "lucide-react";
import { useEffect, useState } from "react";

const API_BASE_URL = "http://localhost:8000";

interface Agent {
  id?: string;
  name: string;
  role: string;
  system_prompt: string;
  model: string;
}

export default function AgentsPage() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [editingAgent, setEditingAgent] = useState<Agent | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);
  const [formData, setFormData] = useState<Agent>({
    name: "",
    role: "",
    system_prompt: "",
    model: "",
  });

  // 获取智能体列表
  const fetchAgents = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/agents`);
      if (res.ok) {
        const data = await res.json();
        setAgents(data);
      }
    } catch (error) {
      console.error("获取智能体失败:", error);
    } finally {
      setIsLoading(false);
    }
  };

  // 获取可用模型
  const fetchModels = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/models`);
      if (res.ok) {
        const data = await res.json();
        const modelList = (data.models || []).map((m: any) => m.name);
        setModels(modelList);
        if (modelList.length > 0 && !formData.model) {
          setFormData((prev) => ({ ...prev, model: modelList[0] }));
        }
      }
    } catch (error) {
      console.error("获取模型列表失败:", error);
    }
  };

  useEffect(() => {
    fetchAgents();
    fetchModels();
  }, []);

  // 重置表单
  const resetForm = () => {
    setFormData({ name: "", role: "", system_prompt: "", model: models[0] || "" });
    setEditingAgent(null);
    setShowForm(false);
  };

  // 创建或更新智能体
  const handleSave = async () => {
    if (!formData.name || !formData.model) {
      alert("请填写名称和选择模型");
      return;
    }

    try {
      if (editingAgent?.id) {
        // 更新现有智能体
        const res = await fetch(`${API_BASE_URL}/api/agents/${editingAgent.id}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(formData),
        });

        if (res.ok) {
          resetForm();
          fetchAgents();
        } else {
          alert("更新失败，请重试");
        }
      } else {
        // 创建新智能体
        const res = await fetch(`${API_BASE_URL}/api/agents`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(formData),
        });

        if (res.ok) {
          resetForm();
          fetchAgents();
        } else {
          alert("创建失败，请检查后端服务是否运行");
        }
      }
    } catch (error) {
      console.error("保存智能体失败:", error);
      alert("操作失败，请检查后端服务是否运行");
    }
  };

  // 编辑智能体
  const handleEdit = (agent: Agent) => {
    setEditingAgent(agent);
    setFormData({
      name: agent.name,
      role: agent.role,
      system_prompt: agent.system_prompt,
      model: agent.model,
    });
    setShowForm(true);
  };

  // 删除智能体
  const handleDelete = async (agentId: string) => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/agents/${agentId}`, {
        method: "DELETE",
      });

      if (res.ok) {
        setDeleteConfirm(null);
        fetchAgents();
      } else {
        alert("删除失败，请重试");
      }
    } catch (error) {
      console.error("删除智能体失败:", error);
      alert("删除失败，请检查后端服务是否运行");
    }
  };

  const handleInputChange = (field: keyof Agent, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 bg-muted/10 p-8">
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">智能体管理</h1>
            <p className="text-muted-foreground">创建和管理您的 AI 智能体</p>
          </div>
          <button
            onClick={() => {
              resetForm();
              setShowForm(!showForm);
            }}
            className="flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-primary-foreground hover:bg-primary/90"
            aria-label="创建智能体"
            tabIndex={0}
          >
            <Plus className="h-4 w-4" />
            创建智能体
          </button>
        </div>

        {/* 创建/编辑表单 */}
        {showForm && (
          <Card className="mb-6">
            <CardHeader>
              <CardTitle>{editingAgent ? "编辑智能体" : "新建智能体"}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div>
                  <label className="mb-2 block text-sm font-medium">名称 *</label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => handleInputChange("name", e.target.value)}
                    className="w-full rounded-lg border bg-background px-3 py-2"
                    placeholder="如：代码助手"
                  />
                </div>
                <div>
                  <label className="mb-2 block text-sm font-medium">角色</label>
                  <input
                    type="text"
                    value={formData.role}
                    onChange={(e) => handleInputChange("role", e.target.value)}
                    className="w-full rounded-lg border bg-background px-3 py-2"
                    placeholder="如：编程专家"
                  />
                </div>
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium">选择模型 *</label>
                <select
                  value={formData.model}
                  onChange={(e) => handleInputChange("model", e.target.value)}
                  className="w-full rounded-lg border bg-background px-3 py-2"
                >
                  {models.length === 0 ? (
                    <option value="">暂无可用模型</option>
                  ) : (
                    models.map((model) => (
                      <option key={model} value={model}>
                        {model}
                      </option>
                    ))
                  )}
                </select>
              </div>
              <div>
                <label className="mb-2 block text-sm font-medium">系统提示词</label>
                <textarea
                  value={formData.system_prompt}
                  onChange={(e) => handleInputChange("system_prompt", e.target.value)}
                  className="w-full rounded-lg border bg-background px-3 py-2"
                  rows={4}
                  placeholder="定义智能体的行为和能力..."
                />
              </div>
              <div className="flex gap-2">
                <button
                  onClick={handleSave}
                  className="rounded-lg bg-primary px-4 py-2 text-primary-foreground hover:bg-primary/90"
                >
                  {editingAgent ? "更新" : "保存"}
                </button>
                <button
                  onClick={resetForm}
                  className="rounded-lg border px-4 py-2 hover:bg-accent"
                >
                  取消
                </button>
              </div>
            </CardContent>
          </Card>
        )}

        {/* 删除确认对话框 */}
        {deleteConfirm && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
            <Card className="w-full max-w-md">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Trash2 className="h-5 w-5 text-destructive" />
                  确认删除
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="mb-4 text-muted-foreground">
                  确定要删除这个智能体吗？此操作无法撤销。
                </p>
                <div className="flex justify-end gap-2">
                  <button
                    onClick={() => setDeleteConfirm(null)}
                    className="rounded-lg border px-4 py-2 hover:bg-accent"
                    tabIndex={0}
                    aria-label="取消删除"
                  >
                    取消
                  </button>
                  <button
                    onClick={() => handleDelete(deleteConfirm)}
                    className="rounded-lg bg-destructive px-4 py-2 text-destructive-foreground hover:bg-destructive/90"
                    tabIndex={0}
                    aria-label="确认删除"
                  >
                    删除
                  </button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* 智能体列表 */}
        {isLoading ? (
          <p className="text-muted-foreground">加载中...</p>
        ) : agents.length === 0 ? (
          <Card>
            <CardContent className="flex flex-col items-center justify-center py-12">
              <Bot className="mb-4 h-12 w-12 text-muted-foreground" />
              <p className="text-lg font-medium">暂无智能体</p>
              <p className="text-muted-foreground">点击上方按钮创建您的第一个智能体</p>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {agents.map((agent) => (
              <Card key={agent.id || agent.name} className="group relative">
                {/* 操作按钮 */}
                <div className="absolute right-2 top-2 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                  <button
                    onClick={() => handleEdit(agent)}
                    className="rounded-lg p-2 hover:bg-accent"
                    aria-label={`编辑 ${agent.name}`}
                    tabIndex={0}
                  >
                    <Pencil className="h-4 w-4" />
                  </button>
                  <button
                    onClick={() => setDeleteConfirm(agent.id || "")}
                    className="rounded-lg p-2 text-destructive hover:bg-destructive/10"
                    aria-label={`删除 ${agent.name}`}
                    tabIndex={0}
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Bot className="h-5 w-5" />
                    {agent.name}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="mb-2 text-sm text-muted-foreground">
                    角色：{agent.role || "未设置"}
                  </p>
                  <p className="mb-2 text-sm text-muted-foreground">
                    模型：{agent.model}
                  </p>
                  <p className="line-clamp-2 text-sm">
                    {agent.system_prompt || "暂无系统提示词"}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
