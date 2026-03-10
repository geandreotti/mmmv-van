import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  turbopack: {
    // Force workspace root to this package to avoid parent lockfile inference.
    root: __dirname,
  },
};

export default nextConfig;
