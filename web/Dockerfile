# Giao diện Next.js (cổng 3000).
#
# BỐN biến, hai nhóm, đều phải có mặt lúc BUILD — và sai theo hai kiểu khác nhau:
#
#   BACKEND_*_ORIGIN   phía server. `rewrites()` trong next.config.mjs được Next
#                      đọc một lần khi `next build` rồi ghi cứng vào
#                      .next/routes-manifest.json. Đặt ở compose `environment:`
#                      thì lúc chạy không còn ai đọc nữa.
#
#   NEXT_PUBLIC_*      phía TRÌNH DUYỆT, nướng thẳng vào bundle. Thiếu chúng thì
#                      lib/api.ts rơi về `?? "http://localhost:8000"` — và
#                      localhost lúc đó là máy của NGƯỜI DÙNG, không phải server.
#                      Container vẫn healthy, curl từ trong máy chủ vẫn 200, chỉ
#                      trình duyệt ở xa là trắng dữ liệu. Đúng kiểu lỗi không lộ
#                      ra ở đâu cả, nên .dockerignore loại .env.local là không đủ
#                      — giá trị phải đi qua ARG.
#
# Cả bốn để mặc định là đường dẫn TƯƠNG ĐỐI hoặc tên service nội bộ, không có IP:
# trình duyệt chỉ gọi về chính cổng 3000 nó đang mở, Next chuyển tiếp nội bộ. Nhờ
# vậy đổi IP máy chủ không phải build lại ảnh.
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci


FROM node:20-alpine AS builder
WORKDIR /app

ARG BACKEND_API_ORIGIN=http://api:8000
ARG BACKEND_LANGGRAPH_ORIGIN=http://chatbot:2024
ARG NEXT_PUBLIC_API_URL=/backend
ARG NEXT_PUBLIC_LANGGRAPH_URL=/langgraph
ENV BACKEND_API_ORIGIN=$BACKEND_API_ORIGIN \
    BACKEND_LANGGRAPH_ORIGIN=$BACKEND_LANGGRAPH_ORIGIN \
    NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL \
    NEXT_PUBLIC_LANGGRAPH_URL=$NEXT_PUBLIC_LANGGRAPH_URL \
    NEXT_TELEMETRY_DISABLED=1

COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build


FROM node:20-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1

COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/next.config.mjs ./next.config.mjs

EXPOSE 3000
CMD ["npm", "run", "start"]
