import { BookIcon, ImageIcon, PlayIcon, SparkIcon } from "./icons";

// Mỗi thẻ một icon riêng: hai thẻ dùng chung SparkIcon thì icon thôi làm nhiệm
// vụ phân loại, chỉ còn là trang trí.
const EXAMPLES = [
  { icon: ImageIcon, label: "Tìm tư liệu", query: "Cho xem ảnh chuyến du lịch Bản Lác – Mai Châu 2024" },
  { icon: PlayIcon, label: "Khám phá video", query: "Video nào có nói về kết nạp thành viên?" },
  { icon: BookIcon, label: "Tra cứu nội quy", query: "Vắng 3 buổi sinh hoạt có bị nhắc nhở không?" },
  { icon: SparkIcon, label: "Tổng hợp sự kiện", query: "Sự kiện gần đây của CLB có những gì?" },
];

export function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col items-center px-1 py-10 text-center">
      <div className="relative mb-6">
        <div className="absolute inset-0 scale-150 rounded-full bg-accent/10 blur-xl" />
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src="/logo.png" alt="HIT MIRA" className="relative h-16 w-16 rounded-2xl border-4 border-white shadow-xl" />
      </div>
      <span className="mb-3 rounded-full border border-accent-ring/40 bg-accent-soft px-3 py-1 text-xs font-semibold text-accent-ink">Trợ lý tri thức đa phương thức</span>
      <h1 className="max-w-[20ch] text-3xl font-semibold leading-tight tracking-[-.035em] text-zinc-900 sm:text-4xl">Khám phá kho tư liệu HIT bằng một câu hỏi</h1>
      <p className="mt-4 max-w-[56ch] text-[15px] leading-7 text-zinc-500">Tìm đúng ảnh, khoảnh khắc trong video và điều khoản nội quy. Mỗi câu trả lời đều đi kèm nguồn để bạn kiểm chứng.</p>
      {/* 4 cột trên màn rộng: vừa dùng hết bề ngang, vừa hạ chiều cao khối chào
          xuống một nửa nên không còn khoảng chết trước ô nhập. */}
      <div className="mt-9 grid w-full grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {EXAMPLES.map(({ icon: Icon, label, query }) => (
          <button key={query} onClick={() => onPick(query)} className="group surface flex items-start gap-3 rounded-2xl p-4 text-left transition-all hover:-translate-y-0.5 hover:border-accent-ring hover:shadow-lg active:translate-y-0">
            <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-accent-soft text-accent"><Icon width={17} height={17} /></span>
            {/* Câu hỏi mới là thứ người ta đọc và bấm, nên nó phải đậm hơn nhãn
                phân loại phía trên chứ không phải ngược lại. */}
            <span><span className="block text-[11px] font-medium uppercase tracking-wide text-zinc-500">{label}</span><span className="mt-1 block text-sm leading-relaxed text-zinc-700 group-hover:text-zinc-900">{query}</span></span>
          </button>
        ))}
      </div>
    </div>
  );
}
