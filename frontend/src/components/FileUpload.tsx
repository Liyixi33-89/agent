"use client";

import { useState, useRef, ChangeEvent } from "react";
import { Upload, File, X, CheckCircle, AlertCircle, Loader2 } from "lucide-react";
import { useAppConfig } from "@/lib/config-context";

interface FileUploadProps {
  onUploadSuccess: (filePath: string) => void;
  accept?: string;
  maxSize?: number; // 最大文件大小（MB）
  className?: string;
}

interface UploadStatus {
  status: "idle" | "uploading" | "success" | "error";
  message?: string;
  filePath?: string;
}

/**
 * 文件上传组件
 * 支持拖拽上传和点击上传
 */
export function FileUpload({
  onUploadSuccess,
  accept = ".csv,.json",
  maxSize = 50,
  className = "",
}: FileUploadProps) {
  const { getApiUrl } = useAppConfig();
  const [isDragging, setIsDragging] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<UploadStatus>({ status: "idle" });
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const handleFileSelect = (file: File) => {
    // 检查文件类型
    const allowedTypes = accept.split(",").map((t) => t.trim().toLowerCase());
    const fileExt = `.${file.name.split(".").pop()?.toLowerCase()}`;
    
    if (!allowedTypes.includes(fileExt)) {
      setUploadStatus({
        status: "error",
        message: `不支持的文件类型。请上传 ${accept} 格式的文件`,
      });
      return;
    }

    // 检查文件大小
    const fileSizeMB = file.size / (1024 * 1024);
    if (fileSizeMB > maxSize) {
      setUploadStatus({
        status: "error",
        message: `文件过大。最大支持 ${maxSize}MB，当前文件 ${fileSizeMB.toFixed(2)}MB`,
      });
      return;
    }

    setSelectedFile(file);
    setUploadStatus({ status: "idle" });
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setUploadStatus({ status: "uploading", message: "正在上传..." });

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(getApiUrl("/api/upload/dataset"), {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `上传失败: HTTP ${response.status}`);
      }

      const result = await response.json();
      
      setUploadStatus({
        status: "success",
        message: "上传成功！",
        filePath: result.file_path,
      });

      // 通知父组件
      onUploadSuccess(result.file_path);

    } catch (error) {
      setUploadStatus({
        status: "error",
        message: error instanceof Error ? error.message : "上传失败，请重试",
      });
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setUploadStatus({ status: "idle" });
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className={`space-y-3 ${className}`}>
      {/* 拖拽区域 */}
      <div
        role="button"
        tabIndex={0}
        aria-label="点击或拖拽上传文件"
        onClick={handleClick}
        onKeyDown={(e) => e.key === "Enter" && handleClick()}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-6 transition-colors ${
          isDragging
            ? "border-primary bg-primary/5"
            : uploadStatus.status === "error"
            ? "border-red-300 bg-red-50"
            : "border-muted-foreground/25 hover:border-primary/50 hover:bg-muted/30"
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={accept}
          onChange={handleFileInputChange}
          className="hidden"
          aria-hidden="true"
        />

        {uploadStatus.status === "uploading" ? (
          <Loader2 className="mb-2 h-8 w-8 animate-spin text-primary" />
        ) : uploadStatus.status === "success" ? (
          <CheckCircle className="mb-2 h-8 w-8 text-green-500" />
        ) : uploadStatus.status === "error" ? (
          <AlertCircle className="mb-2 h-8 w-8 text-red-500" />
        ) : (
          <Upload className="mb-2 h-8 w-8 text-muted-foreground" />
        )}

        <p className="text-sm font-medium">
          {isDragging ? "释放文件" : "点击或拖拽上传数据集"}
        </p>
        <p className="mt-1 text-xs text-muted-foreground">
          支持 {accept} 格式，最大 {maxSize}MB
        </p>
      </div>

      {/* 选中的文件信息 */}
      {selectedFile && (
        <div className="flex items-center justify-between rounded-lg border bg-muted/30 px-3 py-2">
          <div className="flex items-center gap-2">
            <File className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm font-medium">{selectedFile.name}</span>
            <span className="text-xs text-muted-foreground">
              ({(selectedFile.size / 1024).toFixed(1)} KB)
            </span>
          </div>
          <button
            onClick={handleClear}
            className="rounded p-1 hover:bg-muted"
            aria-label="清除选择"
            type="button"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* 状态消息 */}
      {uploadStatus.message && (
        <p
          className={`text-sm ${
            uploadStatus.status === "error"
              ? "text-red-600"
              : uploadStatus.status === "success"
              ? "text-green-600"
              : "text-muted-foreground"
          }`}
        >
          {uploadStatus.message}
        </p>
      )}

      {/* 上传成功后显示文件路径 */}
      {uploadStatus.filePath && (
        <div className="rounded-lg bg-green-50 p-2 text-sm">
          <span className="font-medium text-green-700">文件路径：</span>
          <code className="ml-1 text-green-600">{uploadStatus.filePath}</code>
        </div>
      )}

      {/* 上传按钮 */}
      {selectedFile && uploadStatus.status !== "success" && (
        <button
          onClick={handleUpload}
          disabled={uploadStatus.status === "uploading"}
          className="w-full rounded-lg bg-primary px-4 py-2 text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
          type="button"
        >
          {uploadStatus.status === "uploading" ? "上传中..." : "开始上传"}
        </button>
      )}
    </div>
  );
}
