import { tachKhoi, type Manh } from "@/lib/markdown";

/**
 * Hiện câu trả lời của agent với định dạng markdown tối giản.
 *
 * Trước đây câu trả lời đổ thẳng vào một `<p>` văn bản thuần, nên người dùng
 * đọc thấy nguyên dấu sao: `**Nội Quy CLB 2022.pdf**`. Agent vẫn luôn viết
 * markdown — prompt còn nói rõ là được phép dùng.
 *
 * Dựng bằng phần tử React chứ không phải `dangerouslySetInnerHTML`: nội dung do
 * LLM sinh, nên không được để nó có đường nào chèn HTML vào trang.
 */

function ManhChu({ manh }: { manh: Manh }) {
  switch (manh.loai) {
    case "dam":
      return <strong className="font-semibold text-zinc-900">{manh.chu}</strong>;
    case "ma":
      return (
        <code className="rounded bg-zinc-100 px-1 py-0.5 font-mono text-[.9em] text-zinc-700">
          {manh.chu}
        </code>
      );
    case "lien_ket":
      return (
        <a
          href={manh.href}
          target="_blank"
          rel="noreferrer noopener"
          className="break-all text-accent underline decoration-accent-ring underline-offset-2 hover:text-accent-ink"
        >
          {manh.chu}
        </a>
      );
    default:
      return <>{manh.chu}</>;
  }
}

function Dong({ manh }: { manh: Manh[] }) {
  return (
    <>
      {manh.map((m, i) => (
        <ManhChu key={i} manh={m} />
      ))}
    </>
  );
}

export function RichText({
  text,
  children,
}: {
  text: string;
  /** Con trỏ streaming — phải nằm sau mẩu chữ cuối cùng, không xuống dòng mới. */
  children?: React.ReactNode;
}) {
  const khoi = tachKhoi(text);
  if (khoi.length === 0) return <>{children}</>;

  return (
    // `max-w-[72ch]` là hàng rào cho bề rộng đọc (~75–85 ký tự/dòng). Bong bóng
    // trả lời rộng bằng cả cột để lưới ảnh bên dưới có chỗ, nhưng chữ mà kéo dài
    // theo thì mắt mất dòng khi xuống hàng.
    <div className="max-w-[72ch] space-y-3 text-[15px] leading-7 text-zinc-800">
      {khoi.map((k, i) => {
        const cuoi = i === khoi.length - 1;
        if (k.loai === "danh_sach") {
          const Tag = k.co_so ? "ol" : "ul";
          return (
            <Tag
              key={i}
              className={`space-y-1 pl-5 ${k.co_so ? "list-decimal" : "list-disc"} marker:text-zinc-400`}
            >
              {k.muc.map((m, j) => (
                <li
                  key={j}
                  // Mục con thụt vào và đổi dấu đầu dòng, để "Lần 1: Phạt
                  // 20.000đ" nhìn ra ngay là thuộc về "Đối với cá nhân".
                  className={m.cap === 1 ? "ml-4 list-[circle] text-[.95em]" : ""}
                >
                  <Dong manh={m.noi_dung} />
                  {cuoi && j === k.muc.length - 1 ? children : null}
                </li>
              ))}
            </Tag>
          );
        }
        return (
          <p key={i}>
            {k.dong.map((d, j) => (
              <span key={j}>
                {j > 0 && <br />}
                <Dong manh={d} />
              </span>
            ))}
            {cuoi ? children : null}
          </p>
        );
      })}
    </div>
  );
}
