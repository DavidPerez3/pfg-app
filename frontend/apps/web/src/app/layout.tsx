import type { Metadata } from "next";
import "./globals.css";
import React from "react";
import { NuqsAdapter } from "nuqs/adapters/next/app";
import { SessionProvider } from "next-auth/react";

export const metadata: Metadata = {
  title: "PFG – Recommender AI",
  description: "Sistema de Recomendación con IA y LangGraph",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <SessionProvider>
          <NuqsAdapter>{children}</NuqsAdapter>
        </SessionProvider>
      </body>
    </html>
  );
}
