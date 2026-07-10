import type { Metadata } from "next";
import { Fraunces, Nunito } from "next/font/google";
import "./globals.css";
import { Scene } from "@/components/scene";

const nunito = Nunito({ variable: "--font-nunito", subsets: ["latin"], weight: ["400", "600", "700", "800"] });
const fraunces = Fraunces({ variable: "--font-fraunces", subsets: ["latin"], weight: ["500", "600", "700"] });

export const metadata: Metadata = {
  title: "mini_networks · playground",
  description: "Watch neural networks learn — an enchanted ML playground.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${nunito.variable} ${fraunces.variable} h-full antialiased`}>
      <body className="h-full overflow-hidden">
        <Scene />
        <div className="relative z-10 grid h-full grid-rows-[auto_1fr_auto]">{children}</div>
      </body>
    </html>
  );
}
