/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    // Next.js 15 devtools segment explorer is currently crashing in this
    // workspace on Windows/webpack dev mode. We disable it to stabilize the app
    // runtime while keeping the rest of the dev overlay available.
    devtoolSegmentExplorer: false,
  },
};

export default nextConfig;
