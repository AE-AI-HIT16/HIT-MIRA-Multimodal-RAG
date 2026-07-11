const EXAMPLES = [
  "Cho xem ảnh chuyến du lịch Bản Lác – Mai Châu 2024",
  "Video nào có nói về kết nạp thành viên?",
  "Vắng 3 buổi sinh hoạt có bị nhắc nhở không?",
  "Sự kiện gần đây của CLB có những gì?",
];

export function EmptyState({ onPick }: { onPick: (q: string) => void }) {
  return (
    <div className="mx-auto flex max-w-2xl flex-col items-start px-1 py-16 sm:py-24">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src="/logo.png" alt="HIT" className="mb-5 h-12 w-12 rounded-xl" />
      <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 sm:text-3xl">
        Hỏi về tư liệu &amp; nội quy CLB Tin học HIT
      </h1>
      <p className="mt-2 max-w-[52ch] text-[15px] leading-relaxed text-zinc-500">
        Tìm ảnh, đoạn video theo lời nói, hay tra cứu điều khoản nội quy — mọi câu
        trả lời đều dẫn nguồn, không bịa.
      </p>

      <div className="mt-7 grid w-full grid-cols-1 gap-2 sm:grid-cols-2">
        {EXAMPLES.map((q) => (
          <button
            key={q}
            onClick={() => onPick(q)}
            className="group rounded-xl border border-zinc-200 bg-white px-4 py-3 text-left text-sm text-zinc-600 transition-all hover:border-accent-ring hover:text-zinc-900 active:scale-[0.99]"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
