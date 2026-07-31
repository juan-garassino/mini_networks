import type { Metadata } from "next";
import { Big_Shoulders, Martian_Mono } from "next/font/google";
import "./globals.css";

const display = Big_Shoulders({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["500", "700", "800"],
});
const draft = Martian_Mono({
  variable: "--font-draft",
  subsets: ["latin"],
  weight: ["300", "400", "700"],
});

export const metadata: Metadata = {
  title: "mini_networks · the periodic table of neural networks",
  description: "Atoms, molecules and reactions of the 44-model zoo — drafted as a blueprint.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${display.variable} ${draft.variable} h-full antialiased`}>
      <body className="h-full overflow-hidden">
        <div className="relative z-10 h-full p-3 sm:p-5">
          <div className="bp-frame relative grid h-full grid-rows-[1fr_auto] overflow-hidden">
            {children}
          </div>
        </div>
      </body>
    </html>
  );
}
