export function mmss(sec?: number | null): string {
  if (sec == null || Number.isNaN(sec)) return "00:00";
  const s = Math.max(0, Math.floor(sec));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${String(m).padStart(2, "0")}:${String(r).padStart(2, "0")}`;
}

export const SOURCE_LABEL: Record<string, string> = {
  media: "Ảnh & video",
  regulation: "Nội quy",
  both: "Ảnh, video & nội quy",
};
