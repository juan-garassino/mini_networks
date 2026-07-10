import type { NextConfig } from "next";

// Static export — FastAPI serves the built `out/` at /. No Node at runtime.
const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
};

export default nextConfig;
