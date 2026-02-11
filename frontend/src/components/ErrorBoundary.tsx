"use client";

import { Component, ReactNode, ErrorInfo } from "react";
import { AlertTriangle, RefreshCw, Home } from "lucide-react";
import Link from "next/link";

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

/**
 * 错误边界组件
 * 捕获子组件中的 JavaScript 错误，防止整个应用崩溃
 */
export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
    this.setState({ errorInfo });
    
    // 这里可以将错误上报到日志服务
    // logErrorToService(error, errorInfo);
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  render() {
    if (this.state.hasError) {
      // 如果提供了自定义 fallback，使用它
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // 默认错误 UI
      return (
        <div className="flex min-h-screen items-center justify-center bg-muted/10 p-4">
          <div className="max-w-md rounded-lg border bg-card p-6 shadow-lg">
            <div className="mb-4 flex items-center gap-3">
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-red-100">
                <AlertTriangle className="h-6 w-6 text-red-600" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-foreground">
                  出错了
                </h2>
                <p className="text-sm text-muted-foreground">
                  页面发生了意外错误
                </p>
              </div>
            </div>

            {this.state.error && (
              <div className="mb-4 rounded-lg bg-red-50 p-3">
                <p className="text-sm font-medium text-red-800">
                  {this.state.error.message}
                </p>
                {process.env.NODE_ENV === "development" && this.state.errorInfo && (
                  <details className="mt-2">
                    <summary className="cursor-pointer text-xs text-red-600">
                      查看详细信息
                    </summary>
                    <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap text-xs text-red-700">
                      {this.state.errorInfo.componentStack}
                    </pre>
                  </details>
                )}
              </div>
            )}

            <div className="flex gap-2">
              <button
                onClick={this.handleRetry}
                className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2 text-primary-foreground hover:bg-primary/90"
                type="button"
                aria-label="重试"
              >
                <RefreshCw className="h-4 w-4" />
                重试
              </button>
              <Link
                href="/"
                className="flex flex-1 items-center justify-center gap-2 rounded-lg border px-4 py-2 hover:bg-accent"
                aria-label="返回首页"
              >
                <Home className="h-4 w-4" />
                首页
              </Link>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * 简化版错误边界 - 用于包裹较小的组件区域
 */
interface SimpleErrorBoundaryProps {
  children: ReactNode;
  message?: string;
}

interface SimpleErrorBoundaryState {
  hasError: boolean;
}

export class SimpleErrorBoundary extends Component<
  SimpleErrorBoundaryProps,
  SimpleErrorBoundaryState
> {
  constructor(props: SimpleErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): SimpleErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("SimpleErrorBoundary:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-center">
          <AlertTriangle className="mx-auto mb-2 h-6 w-6 text-red-500" />
          <p className="text-sm text-red-600">
            {this.props.message || "加载组件时出错"}
          </p>
          <button
            onClick={() => this.setState({ hasError: false })}
            className="mt-2 text-xs text-red-500 underline hover:no-underline"
            type="button"
          >
            点击重试
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
