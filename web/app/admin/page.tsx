"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { TopBar } from "@/components/TopBar";
import {
  adminStats,
  evalReport,
  indexStatus,
  me,
  runEval,
  startIndex,
  uploadMedia,
  uploadRegulations,
} from "@/lib/api";
import type {
  AdminStats,
  EvalReport,
  IndexJobStatus,
  RegulationResult,
  UploadResult,
  UserOut,
} from "@/lib/types";

const pct = (x: number) => `${Math.round(x * 100)}%`;

function Section({
  title,
  desc,
  children,
}: {
  title: string;
  desc?: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mt-6 rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm sm:p-7">
      <h2 className="text-lg font-semibold tracking-tight text-zinc-900">{title}</h2>
      {desc && <p className="mt-1 max-w-[60ch] text-sm text-zinc-500">{desc}</p>}
      <div className="mt-5">{children}</div>
    </section>
  );
}

function Field({
  label,
  children,
  hint,
}: {
  label: string;
  children: React.ReactNode;
  hint?: string;
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="text-sm font-medium text-zinc-700">{label}</span>
      {children}
      {hint && <span className="text-xs text-zinc-400">{hint}</span>}
    </label>
  );
}

const inputCls =
  "rounded-lg border border-zinc-200 bg-white px-3 py-2 text-sm outline-none transition-colors focus:border-accent-ring";

function IndexRow({
  label,
  indexed,
  total,
}: {
  label: string;
  indexed: number;
  total: number;
}) {
  const ratio = total > 0 ? indexed / total : 0;
  const done = total > 0 && indexed >= total;
  return (
    <div className="flex items-center gap-4 py-2.5">
      <span className="w-20 shrink-0 text-sm text-zinc-600">{label}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-zinc-100">
        <div
          className={`h-full rounded-full ${done ? "bg-accent" : "bg-accent-ring"}`}
          style={{ width: `${Math.max(ratio * 100, total === 0 ? 0 : 2)}%` }}
        />
      </div>
      <span className="w-24 shrink-0 text-right font-mono text-xs text-zinc-500">
        {indexed} / {total}
      </span>
    </div>
  );
}

export default function AdminPage() {
  const [user, setUser] = useState<UserOut | null | undefined>(undefined);
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [report, setReport] = useState<EvalReport | null>(null);

  useEffect(() => {
    me().then((u) => {
      setUser(u);
      if (u?.role === "admin") {
        adminStats().then(setStats).catch(() => setStats(null));
        evalReport().then(setReport).catch(() => setReport(null));
      }
    });
  }, []);

  if (user === undefined) {
    return (
      <div className="min-h-[100dvh]">
        <TopBar />
        <div className="mx-auto max-w-4xl px-4 py-16">
          <div className="shimmer h-8 w-48 rounded" />
        </div>
      </div>
    );
  }

  if (!user || user.role !== "admin") {
    return (
      <div className="min-h-[100dvh]">
        <TopBar />
        <div className="mx-auto max-w-md px-4 py-24 text-center">
          <h1 className="text-xl font-semibold text-zinc-900">Cần quyền admin</h1>
          <p className="mt-2 text-sm text-zinc-500">
            Trang này chỉ dành cho quản trị viên. Tạo tài khoản admin bằng lệnh{" "}
            <code className="rounded bg-zinc-100 px-1 py-0.5 font-mono text-xs">
              python -m scripts.create_admin
            </code>{" "}
            rồi đăng nhập.
          </p>
          <Link
            href="/login"
            className="mt-5 inline-block rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white"
          >
            Đăng nhập
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-[100dvh]">
      <TopBar />
      <main className="mx-auto max-w-7xl px-4 pb-24 sm:px-6">
        <div className="flex flex-col gap-4 py-10 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h1 className="text-3xl font-semibold tracking-[-.03em] text-zinc-900">
              Bảng điều khiển
            </h1>
            <p className="mt-1 text-sm text-zinc-500">
              Nạp dữ liệu, theo dõi index và đo chất lượng.
            </p>
          </div>
          <Link href="/" className="text-sm text-zinc-500 hover:text-zinc-800">
            ← Về chat
          </Link>
        </div>

        {stats && (
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            {[
              ["Tổng tư liệu", stats.images + stats.videos, `${stats.images} ảnh · ${stats.videos} video`],
              ["Bài viết", stats.posts, "Nguồn nội dung đã nạp"],
              ["Văn bản nội quy", stats.regulations_active, `${stats.rule_chunks} điều khoản`],
              [
                "Tiến độ index",
                `${Math.round(((stats.indexed_images + stats.indexed_videos + stats.indexed_rule_chunks) / Math.max(1, stats.images + stats.videos + stats.rule_chunks)) * 100)}%`,
                "Sẵn sàng tìm kiếm",
              ],
            ].map(([label, value, hint]) => (
              <div key={label} className="surface rounded-2xl p-4 sm:p-5">
                <p className="text-xs font-medium text-zinc-500">{label}</p>
                <p className="mt-2 text-2xl font-semibold tracking-tight text-zinc-900">{value}</p>
                <p className="mt-1 truncate text-[11px] text-zinc-400">{hint}</p>
              </div>
            ))}
          </div>
        )}

        <IndexStatus stats={stats} onReload={() => adminStats().then(setStats)} />
        <UploadMediaForm onDone={() => adminStats().then(setStats)} />
        <UploadRegulationForm onDone={() => adminStats().then(setStats)} />
        <EvalPanel report={report} onReport={setReport} />
      </main>
    </div>
  );
}

const JOB_LABELS: Record<string, string> = {
  media: "Index ảnh",
  videos: "Index video",
  regulations: "Index nội quy",
};

const JOB_STATE_LABELS: Record<string, string> = {
  idle: "chưa chạy",
  running: "đang chạy…",
  done: "xong",
  failed: "lỗi",
};

function IndexStatus({
  stats,
  onReload,
}: {
  stats: AdminStats | null;
  onReload: () => void;
}) {
  const [jobs, setJobs] = useState<IndexJobStatus[]>([]);
  const [error, setError] = useState<string | null>(null);
  const running = jobs.some((j) => j.state === "running");

  // Poll trạng thái job: 4s khi có job chạy; job xong → refresh số liệu index.
  useEffect(() => {
    let cancelled = false;
    const load = () =>
      indexStatus()
        .then((js) => {
          if (cancelled) return;
          setJobs((prev) => {
            const wasRunning = prev.some((j) => j.state === "running");
            const stillRunning = js.some((j) => j.state === "running");
            if (wasRunning && !stillRunning) onReload();
            return js;
          });
        })
        .catch(() => {});
    load();
    const t = setInterval(load, running ? 4000 : 20000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running]);

  async function trigger(target: "media" | "videos" | "regulations") {
    setError(null);
    try {
      const st = await startIndex(target);
      setJobs((prev) => [...prev.filter((j) => j.target !== target), st]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Không chạy được pipeline");
    }
  }

  const failedJob = jobs.find((j) => j.state === "failed");

  return (
    <Section
      title="Trạng thái index"
      desc="Index chạy nền phía server (subprocess, theo NFR không đặt job nặng trên request path). Bấm nút để chạy pipeline cho asset chưa index."
    >
      {!stats ? (
        <div className="shimmer h-24 rounded-xl" />
      ) : (
        <div className="rounded-2xl border border-zinc-100 bg-zinc-50/70 p-5">
          <IndexRow label="Ảnh" indexed={stats.indexed_images} total={stats.images} />
          <IndexRow label="Video" indexed={stats.indexed_videos} total={stats.videos} />
          <IndexRow
            label="Nội quy"
            indexed={stats.indexed_rule_chunks}
            total={stats.rule_chunks}
          />
          <div className="mt-4 flex items-center justify-between border-t border-zinc-100 pt-4">
            <span className="text-xs text-zinc-400">
              {stats.posts} bài · {stats.regulations_active} văn bản nội quy
            </span>
            <button
              onClick={onReload}
              className="rounded-lg border border-zinc-200 px-3 py-1.5 text-xs font-medium text-zinc-600 transition-colors hover:border-accent-ring active:scale-[0.97]"
            >
              Làm mới
            </button>
          </div>
        </div>
      )}

      <div className="mt-4 flex flex-wrap items-center gap-2">
        {(["media", "videos", "regulations"] as const).map((target) => {
          const job = jobs.find((j) => j.target === target);
          const busy = job?.state === "running";
          return (
            <button
              key={target}
              onClick={() => trigger(target)}
              disabled={busy}
              className="rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white transition-all hover:bg-zinc-700 active:scale-[0.99] disabled:opacity-60"
            >
              {JOB_LABELS[target]}
              {job && job.state !== "idle" && (
                <span className="ml-2 text-xs text-zinc-300">
                  {JOB_STATE_LABELS[job.state] ?? job.state}
                </span>
              )}
            </button>
          );
        })}
      </div>
      {error && (
        <p className="mt-3 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>
      )}
      {failedJob && failedJob.log_tail && (
        <pre className="mt-3 overflow-x-auto rounded-xl bg-zinc-900 p-4 font-mono text-xs leading-relaxed text-red-200">
          {failedJob.log_tail}
        </pre>
      )}
    </Section>
  );
}

function ResultLine({ ok, children }: { ok: boolean; children: React.ReactNode }) {
  return (
    <p
      className={`rounded-lg px-3 py-2 text-sm ${
        ok ? "bg-accent-soft text-accent-ink" : "bg-red-50 text-red-700"
      }`}
    >
      {children}
    </p>
  );
}

function UploadMediaForm({ onDone }: { onDone: () => void }) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const form = new FormData(e.currentTarget);
      const res = await uploadMedia(form);
      setResult(res);
      onDone();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Lỗi upload");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Section
      title="Nạp media"
      desc="Ảnh/video + metadata bắt buộc. Cần có consent còn hiệu lực (BR-101), file trùng checksum sẽ bị bỏ qua."
    >
      <form onSubmit={submit} className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="sm:col-span-2">
          <Field label="Tệp ảnh/video" hint="jpg · png · webp · mp4 · mov (chọn nhiều)">
            <input
              type="file"
              name="files"
              multiple
              required
              accept=".jpg,.jpeg,.png,.webp,.mp4,.mov"
              className="text-sm text-zinc-600 file:mr-3 file:rounded-lg file:border-0 file:bg-zinc-900 file:px-3 file:py-2 file:text-sm file:text-white"
            />
          </Field>
        </div>
        <Field label="Link bài gốc">
          <input name="source_url" required className={inputCls} placeholder="https://facebook.com/…" />
        </Field>
        <Field label="Ngày đăng">
          <input type="datetime-local" name="posted_at" required className={inputCls} />
        </Field>
        <Field label="Caption gốc (tùy chọn)">
          <input name="caption_original" className={inputCls} placeholder="Mô tả ngắn" />
        </Field>
        <Field label="Tên sự kiện (tùy chọn)">
          <input name="event_name" className={inputCls} placeholder="Kết nạp 2024" />
        </Field>

        <div className="sm:col-span-2 space-y-3">
          <button
            disabled={busy}
            className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white transition-all hover:bg-accent-ink active:scale-[0.99] disabled:opacity-60"
          >
            {busy ? "Đang nạp…" : "Nạp media"}
          </button>
          {error && <ResultLine ok={false}>{error}</ResultLine>}
          {result && (
            <ResultLine ok>
              Tạo {result.created_ids.length} media
              {result.skipped.length > 0 &&
                ` · bỏ qua ${result.skipped.length} (${result.skipped
                  .map((s) => s.reason)
                  .join(", ")})`}
            </ResultLine>
          )}
        </div>
      </form>
    </Section>
  );
}

function UploadRegulationForm({ onDone }: { onDone: () => void }) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<RegulationResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const res = await uploadRegulations(new FormData(e.currentTarget));
      setResult(res);
      onDone();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Lỗi nạp nội quy");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Section
      title="Nạp nội quy"
      desc="File .md/.txt theo Điều/Khoản. Nạp version mới cùng tiêu đề sẽ lưu trữ bản cũ, chỉ bản mới được phục vụ."
    >
      <form onSubmit={submit} className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <div className="sm:col-span-2">
          <Field label="Tệp nội quy" hint=".md hoặc .txt">
            <input
              type="file"
              name="file"
              required
              accept=".md,.txt"
              className="text-sm text-zinc-600 file:mr-3 file:rounded-lg file:border-0 file:bg-zinc-900 file:px-3 file:py-2 file:text-sm file:text-white"
            />
          </Field>
        </div>
        <Field label="Tiêu đề">
          <input name="title" required className={inputCls} placeholder="Nội quy CLB Tin học HIT" />
        </Field>
        <Field label="Phiên bản">
          <input name="version" defaultValue="1" className={inputCls} />
        </Field>

        <div className="sm:col-span-2 space-y-3">
          <button
            disabled={busy}
            className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white transition-all hover:bg-accent-ink active:scale-[0.99] disabled:opacity-60"
          >
            {busy ? "Đang tách…" : "Nạp nội quy"}
          </button>
          {error && <ResultLine ok={false}>{error}</ResultLine>}
          {result && (
            <ResultLine ok>
              Đã tách {result.n_chunks} điều/khoản (văn bản #{result.regulation_id})
              {result.needs_review && " · có mục cần review"}
            </ResultLine>
          )}
        </div>
      </form>
    </Section>
  );
}

function Metric({
  label,
  value,
  pass,
}: {
  label: string;
  value: string;
  pass?: boolean;
}) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-white p-4">
      <div className="flex items-center justify-between">
        <span className="text-xs text-zinc-500">{label}</span>
        {pass != null && (
          <span
            className={`rounded px-1.5 py-0.5 text-xs font-medium ${
              pass ? "bg-accent-soft text-accent-ink" : "bg-red-50 text-red-600"
            }`}
          >
            {pass ? "đạt" : "chưa đạt"}
          </span>
        )}
      </div>
      <div className="mt-1 font-mono text-2xl font-semibold text-zinc-900">{value}</div>
    </div>
  );
}

function EvalPanel({
  report,
  onReport,
}: {
  report: EvalReport | null;
  onReport: (r: EvalReport) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      onReport(await runEval());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Lỗi chạy đánh giá");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Section
      title="Đánh giá chất lượng"
      desc="Recall@k / MRR / latency / định tuyến trên bộ eval_queries. Cần Qdrant + dữ liệu đã index."
    >
      <button
        onClick={run}
        disabled={busy}
        className="rounded-lg bg-zinc-900 px-4 py-2 text-sm font-medium text-white transition-all hover:bg-zinc-700 active:scale-[0.99] disabled:opacity-60"
      >
        {busy ? "Đang chạy…" : "Chạy đánh giá"}
      </button>
      {error && (
        <p className="mt-3 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">{error}</p>
      )}

      {report && (
        <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Metric
            label={`Recall@5 (mục tiêu ${pct(report.targets.recall ?? 0.8)})`}
            value={pct(report.recall["5"] ?? 0)}
            pass={report.passed["recall@5"]}
          />
          <Metric
            label={`MRR (mục tiêu ${(report.targets.mrr ?? 0.6).toFixed(2)})`}
            value={(report.mrr ?? 0).toFixed(2)}
            pass={report.passed.mrr}
          />
          <Metric
            label="Latency p95"
            value={`${Math.round(report.latency_ms.p95 ?? 0)}ms`}
          />
          <Metric
            label="Định tuyến đúng"
            value={
              report.routing_accuracy != null ? pct(report.routing_accuracy) : "—"
            }
          />
          <Metric
            label="Groundedness (nội quy)"
            value={report.groundedness != null ? pct(report.groundedness) : "—"}
          />
          {report.recall_by_category &&
            Object.entries(report.recall_by_category).map(([cat, v]) => (
              <Metric key={cat} label={`Recall@5 · ${cat}`} value={pct(v)} />
            ))}
          <p className="col-span-2 text-xs text-zinc-400 sm:col-span-4">
            {report.n_queries} truy vấn đánh giá.
            {report.notes ? ` ${report.notes}.` : ""}
          </p>
        </div>
      )}
    </Section>
  );
}
