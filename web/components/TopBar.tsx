"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { me, setToken } from "@/lib/api";
import type { UserOut } from "@/lib/types";
import { UserIcon } from "./icons";

export function TopBar() {
  const [user, setUser] = useState<UserOut | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    me()
      .then(setUser)
      .finally(() => setReady(true));
  }, []);

  function logout() {
    setToken(null);
    setUser(null);
  }

  return (
    <header className="sticky top-0 z-20 border-b border-zinc-200 bg-[var(--bg)]/85 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-3xl items-center justify-between px-4">
        <Link href="/" className="flex items-center gap-2.5">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src="/logo.png" alt="HIT" className="h-7 w-7 rounded-lg" />
          <span className="font-mono text-sm font-bold tracking-tight text-zinc-900">
            HIT<span className="text-accent">·</span>MIRA
          </span>
          <span className="hidden text-xs text-zinc-400 sm:inline">
            Trợ lý đa phương thức
          </span>
        </Link>

        <div className="flex items-center gap-3 text-sm">
          {!ready ? (
            <div className="shimmer h-6 w-20 rounded-full" />
          ) : user ? (
            <>
              {user.role === "admin" && (
                <Link
                  href="/admin"
                  className="text-zinc-500 transition-colors hover:text-accent-ink"
                >
                  Admin
                </Link>
              )}
              <span className="hidden items-center gap-1.5 text-zinc-500 sm:flex">
                <UserIcon width={15} height={15} />
                {user.email}
                {user.role === "admin" && (
                  <span className="rounded bg-accent-soft px-1.5 py-0.5 text-xs font-medium text-accent-ink">
                    admin
                  </span>
                )}
              </span>
              <button
                onClick={logout}
                className="text-zinc-400 transition-colors hover:text-zinc-700"
              >
                Đăng xuất
              </button>
            </>
          ) : (
            <Link
              href="/login"
              className="rounded-lg border border-zinc-200 bg-white px-3 py-1.5 text-zinc-700 transition-colors hover:border-accent-ring"
            >
              Đăng nhập
            </Link>
          )}
        </div>
      </div>
    </header>
  );
}
