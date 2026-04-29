export function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((s, v) => s + v, 0) / arr.length;
}

export function std(arr: number[]): number {
  if (arr.length === 0) return 0;
  const m = mean(arr);
  const variance =
    arr.reduce((s, v) => s + (v - m) * (v - m), 0) / arr.length;
  return Math.sqrt(variance);
}

export function median(arr: number[]): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

export function mode<T extends string | number>(arr: T[]): T | null {
  if (arr.length === 0) return null;
  const counts = new Map<T, number>();
  for (const v of arr) counts.set(v, (counts.get(v) ?? 0) + 1);
  let best: T | null = null;
  let bestCount = -1;
  for (const [k, c] of counts.entries()) {
    if (c > bestCount) {
      best = k;
      bestCount = c;
    }
  }
  return best;
}

export function min(arr: number[]): number {
  return arr.reduce((a, b) => (a < b ? a : b), Infinity);
}
export function max(arr: number[]): number {
  return arr.reduce((a, b) => (a > b ? a : b), -Infinity);
}

export function correlation(x: number[], y: number[]): number {
  const n = Math.min(x.length, y.length);
  if (n === 0) return 0;
  const mx = mean(x);
  const my = mean(y);
  let num = 0;
  let dx = 0;
  let dy = 0;
  for (let i = 0; i < n; i++) {
    const a = x[i] - mx;
    const b = y[i] - my;
    num += a * b;
    dx += a * a;
    dy += b * b;
  }
  const denom = Math.sqrt(dx * dy);
  if (denom === 0) return 0;
  return num / denom;
}

export function sigmoid(z: number): number {
  // Numerically stable
  if (z >= 0) {
    const ez = Math.exp(-z);
    return 1 / (1 + ez);
  }
  const ez = Math.exp(z);
  return ez / (1 + ez);
}

export function softmax(zs: number[]): number[] {
  const m = max(zs);
  const exps = zs.map((z) => Math.exp(z - m));
  const sum = exps.reduce((s, v) => s + v, 0);
  return exps.map((v) => v / sum);
}

export function clip(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

export function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function shuffleIndices(n: number, seed = 42): number[] {
  const idx = Array.from({ length: n }, (_, i) => i);
  let s = seed;
  const rng = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  return idx;
}

export function binCounts(values: number[], bins: number): {
  bin: string;
  count: number;
  start: number;
  end: number;
}[] {
  if (values.length === 0) return [];
  const lo = min(values);
  const hi = max(values);
  if (lo === hi) {
    return [{ bin: lo.toFixed(2), count: values.length, start: lo, end: hi }];
  }
  const width = (hi - lo) / bins;
  const buckets = Array.from({ length: bins }, (_, i) => ({
    bin: `${(lo + i * width).toFixed(2)}`,
    count: 0,
    start: lo + i * width,
    end: lo + (i + 1) * width,
  }));
  for (const v of values) {
    let i = Math.floor((v - lo) / width);
    if (i >= bins) i = bins - 1;
    if (i < 0) i = 0;
    buckets[i].count++;
  }
  return buckets;
}

export function fmt(n: number, digits = 4): string {
  if (!Number.isFinite(n)) return String(n);
  if (Math.abs(n) >= 1000 || (Math.abs(n) < 0.001 && n !== 0)) {
    return n.toExponential(digits - 1);
  }
  return n.toFixed(digits);
}
