import type { Dataset } from "./dataset";
import { mean, median, mode, std } from "./math-utils";

export type ImputeStrategy = "drop" | "mean" | "median" | "mode" | "zero";
export type EncodingStrategy = "label" | "onehot";
export type ScalingStrategy = "none" | "standard" | "minmax";

export interface PreprocessConfig {
  targetColumn: string;
  featureColumns: string[];
  impute: ImputeStrategy;
  encoding: EncodingStrategy;
  scaling: ScalingStrategy;
}

export interface ProcessedDataset {
  featureMatrix: number[][]; // n × d
  featureNames: string[]; // length d
  y: number[]; // length n (encoded labels)
  classNames: string[]; // index = encoded label
  isBinary: boolean;
  /** Per-feature stats used for scaling — needed at prediction time. */
  scalingStats: { mean: number; std: number; min: number; max: number }[];
  /** Maps original feature → expanded indices (for one-hot). Used by prediction UI. */
  inputSchema: PreprocessInputField[];
  notes: string[];
}

export type PreprocessInputField =
  | {
      kind: "numeric";
      original: string;
      outputIndex: number; // index in feature matrix
    }
  | {
      kind: "categorical-onehot";
      original: string;
      categories: string[];
      outputIndices: number[]; // one per category
    }
  | {
      kind: "categorical-label";
      original: string;
      categories: string[];
      outputIndex: number;
    };

/** Snapshot used for "before vs after" panels. */
export interface PreprocessBeforeAfter {
  rowsBefore: number;
  rowsAfter: number;
  missingBefore: number;
  missingAfter: number;
  featureCountBefore: number;
  featureCountAfter: number;
  sampleBefore: Record<string, string | number | null>[];
  sampleAfter: Record<string, number>[];
}

export function preprocess(
  ds: Dataset,
  cfg: PreprocessConfig,
): { processed: ProcessedDataset; beforeAfter: PreprocessBeforeAfter } {
  const notes: string[] = [];
  const targetCol = ds.columns.find((c) => c.name === cfg.targetColumn);
  if (!targetCol) throw new Error(`Target column ${cfg.targetColumn} not found`);

  // --- 1. Determine working rows. Drop rows with missing target. ---
  const rowsBefore = ds.rows.length;
  const missingBefore = ds.columns.reduce((s, c) => s + c.missing, 0);

  let workingRows = ds.rows.filter(
    (r) => r[cfg.targetColumn] !== null && r[cfg.targetColumn] !== undefined,
  );

  // --- 2. Handle missing values for FEATURE columns ---
  const fillers: Record<string, number | string> = {};
  for (const f of cfg.featureColumns) {
    const col = ds.columns.find((c) => c.name === f);
    if (!col) continue;
    if (col.type === "numeric") {
      const present = workingRows
        .map((r) => r[f])
        .filter((v): v is number => typeof v === "number");
      if (present.length === 0) {
        fillers[f] = 0;
        continue;
      }
      if (cfg.impute === "mean") fillers[f] = mean(present);
      else if (cfg.impute === "median") fillers[f] = median(present);
      else if (cfg.impute === "mode") fillers[f] = mode(present) ?? 0;
      else if (cfg.impute === "zero") fillers[f] = 0;
    } else {
      const present = workingRows
        .map((r) => r[f])
        .filter((v): v is string => typeof v === "string");
      const m = mode(present) ?? "missing";
      fillers[f] = m;
    }
  }

  if (cfg.impute === "drop") {
    workingRows = workingRows.filter((r) =>
      cfg.featureColumns.every((f) => r[f] !== null && r[f] !== undefined),
    );
    notes.push(
      `Dropped rows with any missing feature value. ${workingRows.length} of ${rowsBefore} remain.`,
    );
  } else {
    workingRows = workingRows.map((r) => {
      const out = { ...r };
      for (const f of cfg.featureColumns) {
        if (out[f] === null || out[f] === undefined) {
          out[f] = fillers[f];
        }
      }
      return out;
    });
    notes.push(
      `Imputed missing values using "${cfg.impute}" strategy on ${cfg.featureColumns.length} features.`,
    );
  }

  // --- 3. Encode target ---
  let classNames: string[];
  if (targetCol.type === "numeric") {
    const uniq = Array.from(
      new Set(workingRows.map((r) => Number(r[cfg.targetColumn]))),
    ).sort((a, b) => a - b);
    classNames = uniq.map((v) => String(v));
  } else {
    classNames = Array.from(
      new Set(workingRows.map((r) => String(r[cfg.targetColumn]))),
    ).sort();
  }
  const isBinary = classNames.length === 2;
  const yLabels = workingRows.map((r) =>
    classNames.indexOf(String(r[cfg.targetColumn])),
  );
  notes.push(
    `Target "${cfg.targetColumn}" encoded as ${isBinary ? "binary" : "multi-class"} with ${classNames.length} classes.`,
  );

  // --- 4. Encode features ---
  const expandedColumns: { name: string; values: number[] }[] = [];
  const inputSchema: PreprocessInputField[] = [];
  let featureIdx = 0;

  for (const f of cfg.featureColumns) {
    const col = ds.columns.find((c) => c.name === f);
    if (!col) continue;
    if (col.type === "numeric") {
      const values = workingRows.map((r) => Number(r[f]));
      expandedColumns.push({ name: f, values });
      inputSchema.push({
        kind: "numeric",
        original: f,
        outputIndex: featureIdx,
      });
      featureIdx++;
    } else {
      const cats = Array.from(
        new Set(workingRows.map((r) => String(r[f]))),
      ).sort();
      if (cfg.encoding === "label") {
        const values = workingRows.map((r) => cats.indexOf(String(r[f])));
        expandedColumns.push({ name: f, values });
        inputSchema.push({
          kind: "categorical-label",
          original: f,
          categories: cats,
          outputIndex: featureIdx,
        });
        featureIdx++;
      } else {
        const idxs: number[] = [];
        for (const c of cats) {
          const values = workingRows.map((r) =>
            String(r[f]) === c ? 1 : 0,
          );
          expandedColumns.push({ name: `${f}=${c}`, values });
          idxs.push(featureIdx);
          featureIdx++;
        }
        inputSchema.push({
          kind: "categorical-onehot",
          original: f,
          categories: cats,
          outputIndices: idxs,
        });
      }
    }
  }
  notes.push(
    `Encoded features: ${cfg.encoding === "onehot" ? "one-hot" : "label"} for categorical columns.`,
  );

  // --- 5. Scaling ---
  const scalingStats = expandedColumns.map((c) => {
    const m = mean(c.values);
    const s = std(c.values) || 1;
    let mn = Infinity;
    let mx = -Infinity;
    for (const v of c.values) {
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    return { mean: m, std: s, min: mn, max: mx };
  });

  if (cfg.scaling === "standard") {
    expandedColumns.forEach((c, i) => {
      c.values = c.values.map(
        (v) => (v - scalingStats[i].mean) / scalingStats[i].std,
      );
    });
    notes.push("Applied standard scaling: (x - μ) / σ for each feature.");
  } else if (cfg.scaling === "minmax") {
    expandedColumns.forEach((c, i) => {
      const range = scalingStats[i].max - scalingStats[i].min || 1;
      c.values = c.values.map(
        (v) => (v - scalingStats[i].min) / range,
      );
    });
    notes.push(
      "Applied min-max scaling: (x - min) / (max - min) — values now in [0, 1].",
    );
  }

  // --- 6. Build matrix ---
  const n = workingRows.length;
  const d = expandedColumns.length;
  const X: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row = new Array<number>(d);
    for (let j = 0; j < d; j++) row[j] = expandedColumns[j].values[i];
    X.push(row);
  }

  const sampleAfter: Record<string, number>[] = [];
  for (let i = 0; i < Math.min(5, n); i++) {
    const r: Record<string, number> = {};
    for (let j = 0; j < d; j++) r[expandedColumns[j].name] = X[i][j];
    r["__y__"] = yLabels[i];
    sampleAfter.push(r);
  }

  const missingAfter = X.flat().filter(
    (v) => !Number.isFinite(v),
  ).length;

  return {
    processed: {
      featureMatrix: X,
      featureNames: expandedColumns.map((c) => c.name),
      y: yLabels,
      classNames,
      isBinary,
      scalingStats,
      inputSchema,
      notes,
    },
    beforeAfter: {
      rowsBefore,
      rowsAfter: n,
      missingBefore,
      missingAfter,
      featureCountBefore: cfg.featureColumns.length,
      featureCountAfter: d,
      sampleBefore: ds.rows.slice(0, 5),
      sampleAfter,
    },
  };
}

/** Apply the same transformation chain to a single new sample for prediction. */
export function transformSingle(
  inputs: Record<string, string | number>,
  cfg: PreprocessConfig,
  schema: PreprocessInputField[],
  scalingStats: { mean: number; std: number; min: number; max: number }[],
): number[] {
  const out: number[] = new Array(scalingStats.length).fill(0);
  for (const field of schema) {
    if (field.kind === "numeric") {
      out[field.outputIndex] = Number(inputs[field.original] ?? 0);
    } else if (field.kind === "categorical-label") {
      const v = String(inputs[field.original] ?? "");
      out[field.outputIndex] = field.categories.indexOf(v);
      if (out[field.outputIndex] < 0) out[field.outputIndex] = 0;
    } else {
      const v = String(inputs[field.original] ?? "");
      for (let i = 0; i < field.categories.length; i++) {
        out[field.outputIndices[i]] = field.categories[i] === v ? 1 : 0;
      }
    }
  }
  if (cfg.scaling === "standard") {
    for (let i = 0; i < out.length; i++) {
      out[i] = (out[i] - scalingStats[i].mean) / (scalingStats[i].std || 1);
    }
  } else if (cfg.scaling === "minmax") {
    for (let i = 0; i < out.length; i++) {
      const range = scalingStats[i].max - scalingStats[i].min || 1;
      out[i] = (out[i] - scalingStats[i].min) / range;
    }
  }
  return out;
}
