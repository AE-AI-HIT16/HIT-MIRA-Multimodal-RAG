export const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
const ALLOWED = ["image/jpeg", "image/png", "image/webp"];

// Kiểm ảnh phía client (US-502.1 AC-2). Trả thông báo lỗi, hoặc null nếu hợp lệ.
export function validateImage(file: File): string | null {
  if (!ALLOWED.includes(file.type)) return "Chỉ nhận ảnh JPG, PNG hoặc WEBP";
  if (file.size > MAX_IMAGE_BYTES) return "Ảnh quá lớn (tối đa 8MB)";
  return null;
}
