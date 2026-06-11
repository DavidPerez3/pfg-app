/** @type {import('next').NextConfig} */
const backendProxyTarget =
  process.env.NEXT_PUBLIC_BACKEND_URL?.replace(/\/$/, "") ??
  "http://localhost:8000";

const nextConfig = {
  experimental: {
    // Next.js 15 devtools segment explorer is currently crashing in this
    // workspace on Windows/webpack dev mode. We disable it to stabilize the app
    // runtime while keeping the rest of the dev overlay available.
    devtoolSegmentExplorer: false,
  },
  async rewrites() {
    return [
      {
        source: "/backend/:path*",
        destination: `${backendProxyTarget}/:path*`,
      },
    ];
  },
};

export default nextConfig;
