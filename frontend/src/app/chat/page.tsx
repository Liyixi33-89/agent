"use client";

import { Sidebar } from "@/components/Sidebar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Send, Bot, User } from "lucide-react";
import { useEffect, useState, useRef } from "react";

const API_BASE_URL = "http://localhost:8000";

interface Message {
  role: "user" | "assistant";
  content: string;
}

interface Agent {
  name: string;
  role: string;
  system_prompt: string;
  model: string;
}

export default function ChatPage() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);
  const [selectedModel, setSelectedModel] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // 获取智能体和模型
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [agentsRes, modelsRes] = await Promise.all([
          fetch(`${API_BASE_URL}/api/agents`),
          fetch(`${API_BASE_URL}/api/models`),
        ]);

        if (agentsRes.ok) {
          const agentsData = await agentsRes.json();
          setAgents(agentsData);
        }

        if (modelsRes.ok) {
          const modelsData = await modelsRes.json();
          const modelList = (modelsData.models || []).map((m: any) => m.name);
          setModels(modelList);
          if (modelList.length > 0) {
            setSelectedModel(modelList[0]);
          }
        }
      } catch (error) {
        console.error("获取数据失败:", error);
      }
    };

    fetchData();
  }, []);

  // 自动滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // 发送消息
  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const systemPrompt = selectedAgent?.system_prompt || "你是一个有帮助的AI助手。";
      const model = selectedAgent?.model || selectedModel;

      const res = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          messages: [
            { role: "system", content: systemPrompt },
            ...messages,
            userMessage,
          ],
          stream: false,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        const assistantMessage: Message = {
          role: "assistant",
          content: data.message?.content || "无响应",
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } else {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: "请求失败，请检查后端服务和 Ollama 是否运行" },
        ]);
      }
    } catch (error) {
      console.error("发送消息失败:", error);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "连接失败，请检查后端服务是否运行" },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleSelectAgent = (agent: Agent) => {
    setSelectedAgent(agent);
    setSelectedModel(agent.model);
    setMessages([]);
  };

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex flex-1 bg-muted/10">
        {/* 左侧选择区 */}
        <div className="w-64 border-r bg-card p-4">
          <h2 className="mb-4 font-semibold">选择对话模式</h2>

          {/* 直接选择模型 */}
          <div className="mb-6">
            <label className="mb-2 block text-sm font-medium">直接使用模型</label>
            <select
              value={selectedModel}
              onChange={(e) => {
                setSelectedModel(e.target.value);
                setSelectedAgent(null);
              }}
              className="w-full rounded-lg border bg-background px-3 py-2 text-sm"
            >
              {models.length === 0 ? (
                <option value="">暂无模型</option>
              ) : (
                models.map((model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                ))
              )}
            </select>
          </div>

          {/* 选择智能体 */}
          <div>
            <label className="mb-2 block text-sm font-medium">或选择智能体</label>
            <div className="space-y-2">
              {agents.length === 0 ? (
                <p className="text-sm text-muted-foreground">暂无智能体</p>
              ) : (
                agents.map((agent, index) => (
                  <button
                    key={index}
                    onClick={() => handleSelectAgent(agent)}
                    className={`w-full rounded-lg border p-2 text-left text-sm transition-colors hover:bg-accent ${
                      selectedAgent?.name === agent.name ? "border-primary bg-accent" : ""
                    }`}
                    tabIndex={0}
                    aria-label={`选择智能体 ${agent.name}`}
                  >
                    <div className="font-medium">{agent.name}</div>
                    <div className="text-xs text-muted-foreground">{agent.model}</div>
                  </button>
                ))
              )}
            </div>
          </div>
        </div>

        {/* 右侧对话区 */}
        <div className="flex flex-1 flex-col">
          {/* 头部 */}
          <div className="border-b bg-card p-4">
            <h1 className="text-xl font-bold">对话测试</h1>
            <p className="text-sm text-muted-foreground">
              当前：{selectedAgent ? `智能体 - ${selectedAgent.name}` : `模型 - ${selectedModel || "未选择"}`}
            </p>
          </div>

          {/* 消息区 */}
          <div className="flex-1 overflow-y-auto p-4">
            {messages.length === 0 ? (
              <div className="flex h-full items-center justify-center">
                <div className="text-center text-muted-foreground">
                  <Bot className="mx-auto mb-4 h-12 w-12" />
                  <p>开始与 AI 对话</p>
                  <p className="text-sm">在下方输入框输入消息</p>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {messages.map((msg, index) => (
                  <div
                    key={index}
                    className={`flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    {msg.role === "assistant" && (
                      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground">
                        <Bot className="h-4 w-4" />
                      </div>
                    )}
                    <div
                      className={`max-w-[70%] rounded-lg px-4 py-2 ${
                        msg.role === "user"
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted"
                      }`}
                    >
                      <p className="whitespace-pre-wrap">{msg.content}</p>
                    </div>
                    {msg.role === "user" && (
                      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-muted">
                        <User className="h-4 w-4" />
                      </div>
                    )}
                  </div>
                ))}
                {isLoading && (
                  <div className="flex gap-3">
                    <div className="flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground">
                      <Bot className="h-4 w-4" />
                    </div>
                    <div className="rounded-lg bg-muted px-4 py-2">
                      <p className="text-muted-foreground">思考中...</p>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {/* 输入区 */}
          <div className="border-t bg-card p-4">
            <div className="flex gap-2">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                className="flex-1 resize-none rounded-lg border bg-background px-4 py-2"
                placeholder="输入消息... (Enter 发送，Shift+Enter 换行)"
                rows={2}
                disabled={isLoading}
              />
              <button
                onClick={handleSend}
                disabled={isLoading || !input.trim()}
                className="rounded-lg bg-primary px-4 py-2 text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                aria-label="发送消息"
                tabIndex={0}
              >
                <Send className="h-5 w-5" />
              </button>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
