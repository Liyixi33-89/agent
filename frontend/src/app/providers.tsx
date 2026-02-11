"use client";

import { ReactNode } from "react";
import { AppConfigProvider } from "@/lib/config-context";
import { ErrorBoundary } from "@/components/ErrorBoundary";

/**
 * 全局 Providers 包装组件
 * 包含所有需要的 Context Providers 和 ErrorBoundary
 */
export function Providers({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary>
      <AppConfigProvider>{children}</AppConfigProvider>
    </ErrorBoundary>
  );
}
