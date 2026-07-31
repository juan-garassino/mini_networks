// Chart layout data: element symbols, atomic numbers, and family shelves for
// the 16 atoms. Everything else (levels, parents, descriptions) comes live
// from /web/taxonomy — this file only carries the *drafting* of the chart.
export interface AtomSpec { symbol: string; z: number; family: string }

export const ATOMS: Record<string, AtomSpec> = {
  classifier:         { symbol: "Cv", z: 1,  family: "perception" },
  tabular_classifier: { symbol: "Ml", z: 2,  family: "perception" },
  segmentation:       { symbol: "Un", z: 3,  family: "perception" },
  rnn:                { symbol: "Rc", z: 4,  family: "sequence" },
  transformer:        { symbol: "At", z: 5,  family: "sequence" },
  mamba:              { symbol: "Ss", z: 6,  family: "sequence" },
  vae:                { symbol: "Vl", z: 7,  family: "generative" },
  gan:                { symbol: "Ad", z: 8,  family: "generative" },
  diffusion:          { symbol: "Df", z: 9,  family: "generative" },
  pixelcnn:           { symbol: "Ar", z: 10, family: "generative" },
  simclr:             { symbol: "Ct", z: 11, family: "representation" },
  gnn:                { symbol: "Mp", z: 12, family: "structure" },
  nerf:               { symbol: "Nf", z: 13, family: "structure" },
  reinforce:          { symbol: "Pg", z: 14, family: "decision" },
  rl_maze:            { symbol: "Qv", z: 15, family: "decision" },
  alphazero:          { symbol: "Mc", z: 16, family: "decision" },
};

export const FAMILIES = [
  "perception", "sequence", "generative", "representation", "structure", "decision",
] as const;

export const FAMILY_LABELS: Record<string, string> = {
  perception: "I · Perception",
  sequence: "II · Sequence",
  generative: "III · Generative",
  representation: "IV · Representation",
  structure: "V · Structure",
  decision: "VI · Decision",
};

/** Deterministic 2-letter symbol for derived models (first letter + first
 * distinct consonant), uppercased-then-lowercased chemistry-style. */
export function derivedSymbol(name: string): string {
  const clean = name.replace(/[^a-z]/g, "");
  const first = clean[0] ?? "x";
  const rest = clean.slice(1).split("").find((c) => c !== first) ?? "x";
  return first.toUpperCase() + rest;
}

export function symbolFor(name: string): string {
  return ATOMS[name]?.symbol ?? derivedSymbol(name);
}

export const STATUS_INK: Record<string, string> = {
  running: "var(--redline)",
  pending: "var(--redline)",
  dispatched: "var(--redline)",
  done: "var(--line)",
  failed: "#ff7d6b",
  unknown: "var(--ink-dim)",
};
