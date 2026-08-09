/** @type {import('next').NextConfig} */

// Trình duyệt chỉ nói chuyện với chính origin của web (cổng 3000); Next đứng
// giữa chuyển tiếp sang FastAPI và LangGraph. Nhờ vậy hai cổng 8000 và 2024
// không cần mở ra internet, và cũng hết luôn chuyện CORS.
const BACKEND_API = process.env.BACKEND_API_ORIGIN ?? "http://127.0.0.1:8000";
const BACKEND_LANGGRAPH =
  process.env.BACKEND_LANGGRAPH_ORIGIN ?? "http://127.0.0.1:2024";

const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      { source: "/backend/:path*", destination: `${BACKEND_API}/:path*` },
      { source: "/langgraph/:path*", destination: `${BACKEND_LANGGRAPH}/:path*` },
    ];
  },
};

export default nextConfig;
