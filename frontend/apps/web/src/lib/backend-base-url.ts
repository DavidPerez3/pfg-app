export function getBackendBaseUrl() {
  const explicit = process.env.NEXT_PUBLIC_BACKEND_URL?.trim();
  if (explicit) return explicit.replace(/\/$/, "");

  if (typeof window !== "undefined") {
    const { hostname, port } = window.location;
    if (port === "3000") {
      return `http://${hostname}:8000`;
    }
  }

  return "/backend";
}
