"use client";

import { Sidebar } from "@/components/Sidebar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Box,
  Trash2,
  RefreshCw,
  TestTube,
  FileText,
  Download,
  Search,
  CheckCircle,
  XCircle,
  Loader2,
  Copy,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { useEffect, useState, useCallback } from "react";
import { api, LocalModel, PredictResponse, BatchPredictResponse } from "@/lib/api";

// 预定义的标签映射（用户可以自定义）
const DEFAULT_LABEL_MAPS: Record<string, string[]> = {
  sentiment: ["负面 👎", "正面 👍"],
  emotion: ["愤怒", "厌恶", "恐惧", "快乐", "悲伤", "惊讶"],
  binary: ["否", "是"],
};

export default function ModelsPage() {
  const [models, setModels] = useState<LocalModel[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState<LocalModel | null>(null);
  const [testMode, setTestMode] = useState<"single" | "batch">("single");
  const [testInput, setTestInput] = useState("");
  const [batchInput, setBatchInput] = useState("");
  const [testResult, setTestResult] = useState<PredictResponse | null>(null);
  const [batchResults, setBatchResults] = useState<BatchPredictResponse | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [labelMap, setLabelMap] = useState<string>("sentiment");
  const [customLabels, setCustomLabels] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 获取模型列表
  const fetchModels = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await api.models.listLocal();
      setModels(data);
      if (data.length > 0 && !selectedModel) {
        setSelectedModel(data[0]);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "获取模型列表失败");
      console.error("获取模型列表失败:", err);
    } finally {
      setIsLoading(false);
    }
  }, [selectedModel]);

  useEffect(() => {
    fetchModels();
  }, []);

  // 删除模型
  const handleDeleteModel = async (model: LocalModel) => {
    if (!confirm(`确定要删除模型 "${model.name}" 吗？此操作不可恢复。`)) return;

    try {
      await api.models.deleteLocal(model.name);
      setModels((prev) => prev.filter((m) => m.name !== model.name));
      if (selectedModel?.name === model.name) {
        setSelectedModel(models.length > 1 ? models[0] : null);
      }
      alert("模型已删除");
    } catch (err) {
      alert(err instanceof Error ? err.message : "删除失败");
    }
  };

  // 获取标签名称
  const getLabelName = (prediction: number): string => {
    const labels = customLabels
      ? customLabels.split(",").map((l) => l.trim())
      : DEFAULT_LABEL_MAPS[labelMap] || DEFAULT_LABEL_MAPS.sentiment;
    return labels[prediction] || `类别 ${prediction}`;
  };

  // 单条预测
  const handleSinglePredict = async () => {
    if (!selectedModel || !testInput.trim()) {
      alert("请选择模型并输入测试文本");
      return;
    }

    setIsPredicting(true);
    setTestResult(null);
    setError(null);

    try {
      const result = await api.models.predict({
        model_path: selectedModel.path,
        text: testInput.trim(),
        base_model: selectedModel.base_model || "bert-base-uncased",
      });
      setTestResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "预测失败");
    } finally {
      setIsPredicting(false);
    }
  };

  // 批量预测
  const handleBatchPredict = async () => {
    if (!selectedModel || !batchInput.trim()) {
      alert("请选择模型并输入测试文本");
      return;
    }

    const texts = batchInput
      .split("\n")
      .map((t) => t.trim())
      .filter((t) => t.length > 0);

    if (texts.length === 0) {
      alert("请输入至少一条文本（每行一条）");
      return;
    }

    if (texts.length > 100) {
      alert("单次批量预测最多支持 100 条文本");
      return;
    }

    setIsPredicting(true);
    setBatchResults(null);
    setError(null);

    try {
      const result = await api.models.batchPredict({
        texts,
        model_path: selectedModel.path,
        base_model: selectedModel.base_model || "bert-base-uncased",
      });
      setBatchResults(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "批量预测失败");
    } finally {
      setIsPredicting(false);
    }
  };

  // 复制到剪贴板
  const handleCopyResults = () => {
    if (!batchResults) return;

    const text = batchResults.results
      .map((r) => `${r.text}\t${getLabelName(r.prediction)}\t${(r.confidence * 100).toFixed(1)}%`)
      .join("\n");

    navigator.clipboard.writeText(text);
    alert("已复制到剪贴板");
  };

  // 导出 CSV
  const handleExportCsv = () => {
    if (!batchResults) return;

    const headers = "文本,预测结果,置信度,各类别概率";
    const rows = batchResults.results.map((r) => {
      const escapedText = r.text.replace(/"/g, '""');
      const probs = r.probabilities.map((p) => (p * 100).toFixed(2) + "%").join(";");
      return `"${escapedText}","${getLabelName(r.prediction)}",${(r.confidence * 100).toFixed(2)}%,"${probs}"`;
    });

    const csv = [headers, ...rows].join("\n");
    const blob = new Blob(["\ufeff" + csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `predictions_${selectedModel?.name || "model"}_${Date.now()}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  // 格式化时间
  const formatTime = (isoString: string) => {
    return new Date(isoString).toLocaleString("zh-CN");
  };

  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1 bg-muted/10 p-8">
        {/* 页面标题 */}
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">模型管理</h1>
            <p className="text-muted-foreground">
              管理和测试已训练的模型
            </p>
          </div>
          <button
            onClick={fetchModels}
            disabled={isLoading}
            className="flex items-center gap-2 rounded-lg border px-4 py-2 hover:bg-accent disabled:opacity-50"
            aria-label="刷新模型列表"
            tabIndex={0}
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`} />
            刷新
          </button>
        </div>

        {/* 错误提示 */}
        {error && (
          <div className="mb-6 rounded-lg border border-red-200 bg-red-50 p-4 text-red-700">
            <strong>错误：</strong> {error}
          </div>
        )}

        <div className="grid gap-6 lg:grid-cols-3">
          {/* 左侧：模型列表 */}
          <div className="lg:col-span-1">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Box className="h-5 w-5" />
                  本地模型 ({models.length})
                </CardTitle>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <div className="flex items-center justify-center py-8">
                    <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                  </div>
                ) : models.length === 0 ? (
                  <div className="py-8 text-center text-muted-foreground">
                    <Box className="mx-auto mb-4 h-12 w-12 opacity-50" />
                    <p>暂无训练好的模型</p>
                    <p className="mt-1 text-sm">请先进行模型微调</p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {models.map((model) => (
                      <div
                        key={model.name}
                        onClick={() => setSelectedModel(model)}
                        onKeyDown={(e) => e.key === "Enter" && setSelectedModel(model)}
                        role="button"
                        tabIndex={0}
                        aria-label={`选择模型 ${model.name}`}
                        className={`group cursor-pointer rounded-lg border p-3 transition-colors hover:bg-accent ${
                          selectedModel?.name === model.name
                            ? "border-primary bg-accent"
                            : ""
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-medium">{model.name}</span>
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDeleteModel(model);
                            }}
                            className="rounded p-1 text-red-500 opacity-0 transition-opacity hover:bg-red-50 group-hover:opacity-100"
                            aria-label={`删除模型 ${model.name}`}
                            tabIndex={0}
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                        <div className="mt-1 text-xs text-muted-foreground">
                          <p>{model.size_mb} MB</p>
                          {model.base_model && (
                            <p className="truncate">基于: {model.base_model}</p>
                          )}
                          {model.num_labels && (
                            <p>{model.num_labels} 分类</p>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* 右侧：模型测试 */}
          <div className="lg:col-span-2 space-y-6">
            {/* 模型详情 */}
            {selectedModel && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="h-5 w-5" />
                    模型详情
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid gap-4 md:grid-cols-2">
                    <div>
                      <p className="text-sm text-muted-foreground">模型名称</p>
                      <p className="font-medium">{selectedModel.name}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">文件大小</p>
                      <p className="font-medium">{selectedModel.size_mb} MB</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">基础模型</p>
                      <p className="font-medium">{selectedModel.base_model || "未知"}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">分类数</p>
                      <p className="font-medium">{selectedModel.num_labels || "未知"}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">模型路径</p>
                      <p className="font-mono text-sm break-all">{selectedModel.path}</p>
                    </div>
                    <div>
                      <p className="text-sm text-muted-foreground">修改时间</p>
                      <p className="font-medium">{formatTime(selectedModel.modified_at)}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* 模型测试 */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TestTube className="h-5 w-5" />
                  模型测试
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* 测试模式切换 */}
                <div className="flex gap-2">
                  <button
                    onClick={() => setTestMode("single")}
                    className={`flex-1 rounded-lg border px-4 py-2 text-sm font-medium transition-colors ${
                      testMode === "single"
                        ? "border-primary bg-primary text-primary-foreground"
                        : "hover:bg-accent"
                    }`}
                    aria-label="单条预测模式"
                    tabIndex={0}
                  >
                    单条预测
                  </button>
                  <button
                    onClick={() => setTestMode("batch")}
                    className={`flex-1 rounded-lg border px-4 py-2 text-sm font-medium transition-colors ${
                      testMode === "batch"
                        ? "border-primary bg-primary text-primary-foreground"
                        : "hover:bg-accent"
                    }`}
                    aria-label="批量预测模式"
                    tabIndex={0}
                  >
                    批量预测
                  </button>
                </div>

                {/* 高级设置 */}
                <div className="rounded-lg border bg-muted/30">
                  <button
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="flex w-full items-center justify-between p-3 text-sm font-medium"
                    aria-label={showAdvanced ? "收起高级设置" : "展开高级设置"}
                    tabIndex={0}
                  >
                    <span>标签设置</span>
                    {showAdvanced ? (
                      <ChevronUp className="h-4 w-4" />
                    ) : (
                      <ChevronDown className="h-4 w-4" />
                    )}
                  </button>
                  {showAdvanced && (
                    <div className="border-t p-3 space-y-3">
                      <div>
                        <label className="mb-1 block text-sm font-medium">
                          预设标签映射
                        </label>
                        <select
                          value={labelMap}
                          onChange={(e) => setLabelMap(e.target.value)}
                          className="w-full rounded-lg border bg-background px-3 py-2 text-sm"
                          aria-label="选择标签映射"
                        >
                          <option value="sentiment">情感分析 (负面/正面)</option>
                          <option value="binary">二分类 (否/是)</option>
                          <option value="emotion">情感识别 (6类)</option>
                          <option value="custom">自定义标签</option>
                        </select>
                      </div>
                      {labelMap === "custom" && (
                        <div>
                          <label className="mb-1 block text-sm font-medium">
                            自定义标签 (逗号分隔)
                          </label>
                          <input
                            type="text"
                            value={customLabels}
                            onChange={(e) => setCustomLabels(e.target.value)}
                            placeholder="标签0, 标签1, 标签2..."
                            className="w-full rounded-lg border bg-background px-3 py-2 text-sm"
                          />
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {/* 单条预测 */}
                {testMode === "single" && (
                  <div className="space-y-4">
                    <div>
                      <label className="mb-2 block text-sm font-medium">
                        输入文本
                      </label>
                      <textarea
                        value={testInput}
                        onChange={(e) => setTestInput(e.target.value)}
                        placeholder="输入要预测的文本..."
                        className="w-full resize-none rounded-lg border bg-background px-4 py-3"
                        rows={3}
                        aria-label="输入预测文本"
                      />
                    </div>
                    <button
                      onClick={handleSinglePredict}
                      disabled={isPredicting || !selectedModel || !testInput.trim()}
                      className="flex w-full items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2 text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                      aria-label="执行预测"
                      tabIndex={0}
                    >
                      {isPredicting ? (
                        <>
                          <Loader2 className="h-4 w-4 animate-spin" />
                          预测中...
                        </>
                      ) : (
                        <>
                          <Search className="h-4 w-4" />
                          预测
                        </>
                      )}
                    </button>

                    {/* 单条预测结果 */}
                    {testResult && (
                      <div className="rounded-lg border bg-muted/30 p-4 space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium">预测结果</span>
                          <span
                            className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-medium ${
                              testResult.prediction === 1
                                ? "bg-green-100 text-green-700"
                                : "bg-red-100 text-red-700"
                            }`}
                          >
                            {getLabelName(testResult.prediction)}
                          </span>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">置信度</p>
                          <div className="mt-1 flex items-center gap-2">
                            <div className="h-2 flex-1 rounded-full bg-muted overflow-hidden">
                              <div
                                className={`h-full transition-all ${
                                  testResult.confidence > 0.8
                                    ? "bg-green-500"
                                    : testResult.confidence > 0.6
                                    ? "bg-yellow-500"
                                    : "bg-red-500"
                                }`}
                                style={{ width: `${testResult.confidence * 100}%` }}
                              />
                            </div>
                            <span className="text-sm font-medium">
                              {(testResult.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                        {testResult.probabilities && (
                          <div>
                            <p className="text-sm text-muted-foreground mb-2">各类别概率</p>
                            <div className="space-y-1">
                              {testResult.probabilities.map((prob, idx) => (
                                <div key={idx} className="flex items-center gap-2 text-sm">
                                  <span className="w-20 truncate">{getLabelName(idx)}:</span>
                                  <div className="h-1.5 flex-1 rounded-full bg-muted overflow-hidden">
                                    <div
                                      className="h-full bg-primary"
                                      style={{ width: `${prob * 100}%` }}
                                    />
                                  </div>
                                  <span className="w-14 text-right font-mono">
                                    {(prob * 100).toFixed(2)}%
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* 批量预测 */}
                {testMode === "batch" && (
                  <div className="space-y-4">
                    <div>
                      <label className="mb-2 block text-sm font-medium">
                        输入文本 (每行一条，最多 100 条)
                      </label>
                      <textarea
                        value={batchInput}
                        onChange={(e) => setBatchInput(e.target.value)}
                        placeholder={`输入要预测的文本，每行一条...\n例如:\n这个产品非常好用\n服务态度太差了\n质量一般般`}
                        className="w-full resize-none rounded-lg border bg-background px-4 py-3 font-mono text-sm"
                        rows={6}
                        aria-label="输入批量预测文本"
                      />
                      <p className="mt-1 text-xs text-muted-foreground">
                        当前: {batchInput.split("\n").filter((t) => t.trim()).length} 条
                      </p>
                    </div>
                    <button
                      onClick={handleBatchPredict}
                      disabled={isPredicting || !selectedModel || !batchInput.trim()}
                      className="flex w-full items-center justify-center gap-2 rounded-lg bg-primary px-4 py-2 text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                      aria-label="执行批量预测"
                      tabIndex={0}
                    >
                      {isPredicting ? (
                        <>
                          <Loader2 className="h-4 w-4 animate-spin" />
                          批量预测中...
                        </>
                      ) : (
                        <>
                          <Search className="h-4 w-4" />
                          批量预测
                        </>
                      )}
                    </button>

                    {/* 批量预测结果 */}
                    {batchResults && (
                      <div className="space-y-4">
                        {/* 统计信息 */}
                        <div className="flex items-center justify-between rounded-lg border bg-muted/30 p-3">
                          <div className="flex items-center gap-4">
                            <span className="text-sm">
                              共 <strong>{batchResults.total}</strong> 条结果
                            </span>
                            {(() => {
                              const positive = batchResults.results.filter(
                                (r) => r.prediction === 1
                              ).length;
                              const negative = batchResults.total - positive;
                              return (
                                <>
                                  <span className="flex items-center gap-1 text-sm text-green-600">
                                    <CheckCircle className="h-4 w-4" />
                                    {positive}
                                  </span>
                                  <span className="flex items-center gap-1 text-sm text-red-600">
                                    <XCircle className="h-4 w-4" />
                                    {negative}
                                  </span>
                                </>
                              );
                            })()}
                          </div>
                          <div className="flex gap-2">
                            <button
                              onClick={handleCopyResults}
                              className="flex items-center gap-1 rounded border px-2 py-1 text-xs hover:bg-accent"
                              aria-label="复制结果"
                              tabIndex={0}
                            >
                              <Copy className="h-3 w-3" />
                              复制
                            </button>
                            <button
                              onClick={handleExportCsv}
                              className="flex items-center gap-1 rounded border px-2 py-1 text-xs hover:bg-accent"
                              aria-label="导出 CSV"
                              tabIndex={0}
                            >
                              <Download className="h-3 w-3" />
                              导出 CSV
                            </button>
                          </div>
                        </div>

                        {/* 结果列表 */}
                        <div className="max-h-80 overflow-y-auto rounded-lg border">
                          <table className="w-full text-sm">
                            <thead className="sticky top-0 bg-muted">
                              <tr>
                                <th className="px-3 py-2 text-left font-medium">#</th>
                                <th className="px-3 py-2 text-left font-medium">文本</th>
                                <th className="px-3 py-2 text-left font-medium">预测</th>
                                <th className="px-3 py-2 text-left font-medium">置信度</th>
                              </tr>
                            </thead>
                            <tbody>
                              {batchResults.results.map((result, idx) => (
                                <tr
                                  key={idx}
                                  className="border-t hover:bg-muted/30"
                                >
                                  <td className="px-3 py-2 text-muted-foreground">
                                    {idx + 1}
                                  </td>
                                  <td className="max-w-xs truncate px-3 py-2" title={result.text}>
                                    {result.text}
                                  </td>
                                  <td className="px-3 py-2">
                                    <span
                                      className={`inline-flex rounded-full px-2 py-0.5 text-xs font-medium ${
                                        result.prediction === 1
                                          ? "bg-green-100 text-green-700"
                                          : "bg-red-100 text-red-700"
                                      }`}
                                    >
                                      {getLabelName(result.prediction)}
                                    </span>
                                  </td>
                                  <td className="px-3 py-2 font-mono">
                                    {(result.confidence * 100).toFixed(1)}%
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* 无模型提示 */}
                {!selectedModel && models.length === 0 && (
                  <div className="py-8 text-center text-muted-foreground">
                    <p>请先训练一个模型后再进行测试</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}
