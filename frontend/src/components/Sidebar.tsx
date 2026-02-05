import Link from "next/link";
import { LayoutDashboard, Bot, MessageSquare, Settings, Hammer } from "lucide-react";

const navItems = [
  { name: "仪表盘", href: "/", icon: LayoutDashboard },
  { name: "智能体管理", href: "/agents", icon: Bot },
  { name: "对话测试", href: "/chat", icon: MessageSquare },
  { name: "模型微调", href: "/finetune", icon: Hammer },
  { name: "系统设置", href: "/settings", icon: Settings },
];

export function Sidebar() {
  return (
    <div className="flex h-screen w-64 flex-col border-r bg-card px-4 py-6">
      <div className="mb-8 flex items-center gap-2 px-2">
        <Bot className="h-8 w-8 text-primary" />
        <span className="text-xl font-bold">Agent 微调平台</span>
      </div>
      <nav className="flex flex-1 flex-col gap-2">
        {navItems.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className="flex items-center gap-3 rounded-lg px-3 py-2 text-muted-foreground transition-colors hover:bg-accent hover:text-accent-foreground"
          >
            <item.icon className="h-5 w-5" />
            <span>{item.name}</span>
          </Link>
        ))}
      </nav>
    </div>
  );
}