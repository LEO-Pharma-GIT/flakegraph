import type { Metadata } from "next";
import type { ReactNode } from "react";
// Shipped with the package rather than fetched from Google at build time: the
// image is built on hosts whose egress is intercepted, and a build must not
// depend on reaching a font CDN.
import { GeistMono } from "geist/font/mono";
import { GeistSans } from "geist/font/sans";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "FlakeGraph",
    template: "%s · FlakeGraph",
  },
  description: "Operator control plane for local, Kubernetes, and Snowflake graph runs.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable} h-full antialiased`}>
      <body className="min-h-full bg-background text-foreground">{children}</body>
    </html>
  );
}
