import type { Metadata } from "next";
import {
  Big_Shoulders, Cormorant_Garamond, Instrument_Serif, Jost, Martian_Mono, VT323,
} from "next/font/google";
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
const instrument = Instrument_Serif({ variable: "--font-instrument", subsets: ["latin"], weight: "400" });
const vt = VT323({ variable: "--font-vt", subsets: ["latin"], weight: "400" });
const jost = Jost({ variable: "--font-jost", subsets: ["latin"], weight: ["400", "600", "700"] });
const cormorant = Cormorant_Garamond({ variable: "--font-cormorant", subsets: ["latin"], weight: ["400", "600", "700"] });

export const metadata: Metadata = {
  title: "mini_networks · the periodic table of neural networks",
  description: "Atoms, molecules and reactions of the 44-model zoo — drafted as a blueprint.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${display.variable} ${draft.variable} ${instrument.variable} ${vt.variable} ${jost.variable} ${cormorant.variable} h-full antialiased`}>
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
