export function getBackendBaseUrl() {
  const explicit = process.env.NEXT_PUBLIC_BACKEND_URL?.trim();
  if (explicit) return explicit.replace(/\/$/, "");
  return "/backend";
}
