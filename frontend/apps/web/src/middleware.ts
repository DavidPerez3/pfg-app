import { auth } from "@/auth";
import { NextResponse } from "next/server";

export default auth((req) => {
  if (process.env.E2E_BYPASS_AUTH === "true") {
    return NextResponse.next();
  }
  // If there is no active session, redirect to the login page
  if (!req.auth) {
    const loginUrl = new URL("/login", req.url);
    return NextResponse.redirect(loginUrl);
  }
});

export const config = {
  // Protect all routes EXCEPT the login page and next-auth API endpoints
  matcher: ["/((?!login|api/auth|_next/static|_next/image|favicon.ico).*)"],
};
