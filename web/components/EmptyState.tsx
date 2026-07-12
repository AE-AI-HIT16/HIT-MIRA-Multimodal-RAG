import { BookIcon, ImageIcon, SparkIcon } from "./icons";

const EXAMPLES = [
  { icon: ImageIcon, label: "Tìm tư liệu", query: "Cho xem ảnh chuyến du lịch Bản Lác – Mai Châu 2024" },
  { icon: SparkIcon, label: "Khám phá video", query: "Video nào có nói về kết nạp thành viên?" },
  { icon: BookIcon, label: "Tra cứu nội quy", query: "Vắng 3 buổi sinh hoạt có bị nhắc nhở không?" },
  { icon: SparkIcon, label: "Tổng hợp sự kiện", query: "Sự kiện gần đây của CLB có những gì?" },
];

export function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="mx-auto flex max-w-3xl flex-col items-center px-1 py-14 text-center sm:py-24">
      <div className="relative mb-6">
        <div className="absolute inset-0 scale-150 rounded-full bg-accent/10 blur-xl" />
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src="/logo.png" alt="HIT MIRA" className="relative h-16 w-16 rounded-2xl border-4 border-white shadow-xl" />
      </div>
      <span className="mb-3 rounded-full border border-accent-ring/40 bg-accent-soft px-3 py-1 text-xs font-semibold text-accent-ink">Trợ lý tri thức đa phương thức</span>
      <h1 className="max-w-[20ch] text-3xl font-semibold leading-tight tracking-[-.035em] text-zinc-900 sm:text-4xl">Khám phá kho tư liệu HIT bằng một câu hỏi</h1>
      <p className="mt-4 max-w-[56ch] text-[15px] leading-7 text-zinc-500">Tìm đúng ảnh, khoảnh khắc trong video và điều khoản nội quy. Mỗi câu trả lời đều đi kèm nguồn để bạn kiểm chứng.</p>
      <div className="mt-9 grid w-full grid-cols-1 gap-3 sm:grid-cols-2">
        {EXAMPLES.map(({ icon: Icon, label, query }) => (
          <button key={query} onClick={() => onPick(query)} className="group surface flex items-start gap-3 rounded-2xl p-4 text-left transition-all hover:-translate-y-0.5 hover:border-accent-ring hover:shadow-lg active:translate-y-0">
            <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-accent-soft text-accent"><Icon width={17} height={17} /></span>
            <span><span className="block text-xs font-semibold text-zinc-900">{label}</span><span className="mt-1 block text-sm leading-relaxed text-zinc-500 group-hover:text-zinc-700">{query}</span></span>
          </button>
        ))}
      </div>
      <div className="mt-8 flex flex-wrap justify-center gap-x-5 gap-y-2 text-xs text-zinc-400"><span>Ảnh & video</span><span>Tra cứu nội quy</span><span>Trích dẫn rõ ràng</span></div>
    </div>
  );
}
