"use client";

import { createContext, useContext, useState, useEffect, ReactNode } from "react";

// 配置类型定义
interface AppConfig {
  // 服务配置
  backendUrl: string;
  ollamaUrl: string;
  // 微调默认配置
  defaultBaseModel: string;
  modelSavePath: string;
  defaultBatchSize: number;
  defaultMaxLength: number;
  defaultEpochs: number;
  defaultLearningRate: number;
  useGpuByDefault: boolean;
  // 聊天配置
  currentSessionId: string | null;
}

// 默认配置
const defaultConfig: AppConfig = {
  backendUrl: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  ollamaUrl: "http://localhost:11434",
  defaultBaseModel: "bert-base-uncased",
  modelSavePath: "./models",
  defaultBatchSize: 8,
  defaultMaxLength: 128,
  defaultEpochs: 3,
  defaultLearningRate: 2e-5,
  useGpuByDefault: true,
  currentSessionId: null,
};

// Context 类型
interface AppConfigContextType {
  config: AppConfig;
  updateConfig: (updates: Partial<AppConfig>) => void;
  resetConfig: () => void;
  // 便捷方法
  getApiUrl: (endpoint: string) => string;
  generateSessionId: () => string;
}

// 创建 Context
const AppConfigContext = createContext<AppConfigContextType | undefined>(undefined);

// Provider 组件
export function AppConfigProvider({ children }: { children: ReactNode }) {
  const [config, setConfig] = useState<AppConfig>(defaultConfig);
  const [isLoaded, setIsLoaded] = useState(false);

  // 从 localStorage 加载配置
  useEffect(() => {
    try {
      const savedConfig = localStorage.getItem("appConfig");
      if (savedConfig) {
        const parsed = JSON.parse(savedConfig);
        setConfig((prev) => ({ ...prev, ...parsed }));
      }
    } catch (error) {
      console.error("加载配置失败:", error);
    }
    setIsLoaded(true);
  }, []);

  // 配置变化时保存到 localStorage
  useEffect(() => {
    if (isLoaded) {
      try {
        localStorage.setItem("appConfig", JSON.stringify(config));
      } catch (error) {
        console.error("保存配置失败:", error);
      }
    }
  }, [config, isLoaded]);

  // 更新配置
  const updateConfig = (updates: Partial<AppConfig>) => {
    setConfig((prev) => ({ ...prev, ...updates }));
  };

  // 重置配置
  const resetConfig = () => {
    setConfig(defaultConfig);
    localStorage.removeItem("appConfig");
  };

  // 获取 API URL
  const getApiUrl = (endpoint: string): string => {
    const base = config.backendUrl.replace(/\/$/, "");
    const path = endpoint.startsWith("/") ? endpoint : `/${endpoint}`;
    return `${base}${path}`;
  };

  // 生成会话 ID
  const generateSessionId = (): string => {
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).substring(2, 8);
    return `session_${timestamp}_${random}`;
  };

  const value: AppConfigContextType = {
    config,
    updateConfig,
    resetConfig,
    getApiUrl,
    generateSessionId,
  };

  return (
    <AppConfigContext.Provider value={value}>
      {children}
    </AppConfigContext.Provider>
  );
}

// 自定义 Hook
export function useAppConfig() {
  const context = useContext(AppConfigContext);
  if (context === undefined) {
    throw new Error("useAppConfig must be used within an AppConfigProvider");
  }
  return context;
}

// 导出类型
export type { AppConfig, AppConfigContextType };
