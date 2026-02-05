"use client";

import { Sidebar } from "@/components/Sidebar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Activity, Users, HardDrive, Zap } from "lucide-react";
import { useEffect, useState } from "react";

// API 基础地址
const API_BASE_URL = "http://localhost:8000";

interface DashboardCardProps {
  title: string;
  value: string | number;
  icon: React.ElementType;
  description: string;
}

function DashboardCard({ title, value, icon: Icon, description }: DashboardCardProps) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        <p className="text-xs text-muted-foreground">{description}</p>
      </CardContent>
    </Card>
  );
}

export default function Home() {
  const [agentCount, setAgentCount] = useState(0);
  const [modelCount, setModelCount] = useState(0);
  const [modelNames, setModelNames] = useState("");
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // 获取 Agent 数量
    const fetchAgents = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/agents`);
        if (res.ok) {
          const data = await res.json();
          setAgentCount(data.length);
          setIsConnected(true);
        }
      } catch (error) {
        console.error("无法连接后端服务:", error);
        setIsConnected(false);
      }
    };

    // 获取模型列表
    const fetchModels = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/models`);
        if (res.ok) {
          const data = await res.json();
          const models = data.models || [];
          setModelCount(models.length);
          setModelNames(models.slice(0, 3).map((m: any) => m.name).join(", ") || "暂无模型");
        }
      } catch (error) {
        console.error("获取模型列表失败:", error);
      }
    };

    fetchAgents();
    fetchModels();
  }, []);

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 bg-muted/10 p-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold tracking-tight">仪表盘</h1>
          <p className="text-muted-foreground">
            智能体生态系统概览
            {!isConnected && (
              <span className="ml-2 text-red-500">（后端服务未连接）</span>
            )}
          </p>
        </div>

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <DashboardCard
            title="智能体总数"
            value={agentCount}
            icon={Users}
            description="已配置的智能体数量"
          />
          <DashboardCard
            title="可用模型"
            value={modelCount}
            icon={HardDrive}
            description={modelNames || "暂无模型"}
          />
          <DashboardCard
            title="微调任务"
            value="0"
            icon={Activity}
            description="进行中的微调任务"
          />
          <DashboardCard
            title="API 请求"
            value="-"
            icon={Zap}
            description="统计功能开发中"
          />
        </div>
      </main>
    </div>
  );
}