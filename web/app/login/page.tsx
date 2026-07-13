"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { login, register, setToken } from "@/lib/api";

export default function LoginPage() {
  const router = useRouter();
  const [tab, setTab] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (tab === "register") {
        await register(email, password, name);
      }
      const { access_token } = await login(email, password);
      setToken(access_token);
      router.push("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Không đăng nhập được");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="grid min-h-[100dvh] grid-cols-1 lg:grid-cols-[1.1fr_1fr]">
      {/* Cột thương hiệu — trái, không center hero */}
      <aside className="hidden flex-col justify-between bg-zinc-900 p-12 text-zinc-100 lg:flex">
        <div className="flex items-center gap-2 font-mono text-sm font-bold">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/logo.png" alt="HIT" className="h-6 w-6 rounded-md" />
          HIT<span className="text-accent-ring">·</span>MIRA
        </div>
        <div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/logo.png" alt="CLB Tin học HIT" className="mb-6 h-14 w-14 rounded-2xl" />
          <h2 className="max-w-[18ch] text-3xl font-semibold leading-tight tracking-tight">
            Trợ lý đa phương thức cho tư liệu &amp; nội quy CLB.
          </h2>
          <p className="mt-3 max-w-[42ch] text-sm leading-relaxed text-zinc-400">
            Đăng nhập để lưu lịch sử hội thoại. Chưa có tài khoản vẫn dùng được với
            tư cách khách.
          </p>
        </div>
        <p className="font-mono text-xs text-zinc-600">CLB Tin học HIT · 2026</p>
      </aside>

      {/* Cột form */}
      <main className="flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-sm">
          <div className="mb-6 flex gap-1 rounded-lg bg-zinc-100 p-1 text-sm">
            {(["login", "register"] as const).map((t) => (
              <button
                key={t}
                onClick={() => {
                  setTab(t);
                  setError(null);
                }}
                className={`flex-1 rounded-md py-1.5 font-medium transition-colors ${
                  tab === t ? "bg-white text-zinc-900 shadow-sm" : "text-zinc-500"
                }`}
              >
                {t === "login" ? "Đăng nhập" : "Tạo tài khoản"}
              </button>
            ))}
          </div>

          <form onSubmit={submit} className="space-y-4">
            {tab === "register" && (
              <div className="flex flex-col gap-2">
                <label htmlFor="name" className="text-sm font-medium text-zinc-700">
                  Tên hiển thị
                </label>
                <input
                  id="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  className="rounded-xl border border-zinc-200 bg-white px-3.5 py-2.5 shadow-sm text-sm outline-none transition-colors focus:border-accent-ring"
                  placeholder="Nguyễn Văn A"
                />
              </div>
            )}

            <div className="flex flex-col gap-2">
              <label htmlFor="email" className="text-sm font-medium text-zinc-700">
                Email
              </label>
              <input
                id="email"
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="rounded-xl border border-zinc-200 bg-white px-3.5 py-2.5 shadow-sm text-sm outline-none transition-colors focus:border-accent-ring"
                placeholder="ban@hit.edu.vn"
              />
            </div>

            <div className="flex flex-col gap-2">
              <label htmlFor="password" className="text-sm font-medium text-zinc-700">
                Mật khẩu
              </label>
              <input
                id="password"
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="rounded-xl border border-zinc-200 bg-white px-3.5 py-2.5 shadow-sm text-sm outline-none transition-colors focus:border-accent-ring"
                placeholder="••••••••"
              />
            </div>

            {error && (
              <p className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>
            )}

            <button
              type="submit"
              disabled={busy}
              className="w-full rounded-xl bg-accent py-3 shadow-sm text-sm font-medium text-white transition-all hover:bg-accent-ink active:scale-[0.99] disabled:opacity-60"
            >
              {busy ? "Đang xử lý…" : tab === "login" ? "Đăng nhập" : "Tạo tài khoản"}
            </button>
          </form>

          <Link
            href="/"
            className="mt-4 block text-center text-sm text-zinc-500 transition-colors hover:text-zinc-800"
          >
            Tiếp tục với tư cách khách
          </Link>
        </div>
      </main>
    </div>
  );
}
