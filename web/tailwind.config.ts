import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-sans)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      colors: {
        // Accent: cam thương hiệu HIT (#f06c25) — 1 accent duy nhất.
        accent: {
          DEFAULT: "#f06c25",
          soft: "#fef1e9",
          ring: "#f7b58c",
          ink: "#b8480f",
        },
      },
      keyframes: {
        "msg-in": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        blink: { "0%, 80%, 100%": { opacity: "0.2" }, "40%": { opacity: "1" } },
      },
      animation: {
        "msg-in": "msg-in 0.35s cubic-bezier(0.16,1,0.3,1) both",
        shimmer: "shimmer 1.6s linear infinite",
      },
    },
  },
  plugins: [],
};

export default config;
