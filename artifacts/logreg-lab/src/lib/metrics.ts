import type { LogRegModel } from "./logreg";
import { predictClass, predictProbaBinary, predictProbaOvR } from "./logreg";

export interface ConfusionMatrix {
  classNames: string[];
  matrix: number[][]; // rows = actual, cols = predicted
}

export interface ClassMetrics {
  className: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface EvaluationResult {
  accuracy: number;
  confusion: ConfusionMatrix;
  perClass: ClassMetrics[];
  macroF1: number;
  rocPoints?: { fpr: number; tpr: number; threshold: number }[];
  auc?: number;
}

export function evaluate(
  model: LogRegModel,
  X: number[][],
  y: number[],
  threshold: number,
): EvaluationResult {
  const classNames = model.classNames;
  const k = classNames.length;
  const matrix: number[][] = Array.from({ length: k }, () =>
    new Array(k).fill(0),
  );

  let correct = 0;
  for (let i = 0; i < X.length; i++) {
    const { label } = predictClass(model, X[i], threshold);
    matrix[y[i]][label]++;
    if (label === y[i]) correct++;
  }
  const accuracy = X.length > 0 ? correct / X.length : 0;

  const perClass: ClassMetrics[] = [];
  for (let c = 0; c < k; c++) {
    const tp = matrix[c][c];
    let fp = 0;
    let fn = 0;
    let support = 0;
    for (let r = 0; r < k; r++) {
      support += matrix[c][r];
      if (r !== c) {
        fn += matrix[c][r];
        fp += matrix[r][c];
      }
    }
    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
    const f1 =
      precision + recall === 0
        ? 0
        : (2 * precision * recall) / (precision + recall);
    perClass.push({ className: classNames[c], precision, recall, f1, support });
  }
  const macroF1 = perClass.reduce((s, c) => s + c.f1, 0) / Math.max(1, k);

  const result: EvaluationResult = {
    accuracy,
    confusion: { classNames: [...classNames], matrix },
    perClass,
    macroF1,
  };

  if (model.type === "binary") {
    // Build ROC by sweeping threshold
    const probs: { p: number; y: number }[] = X.map((row, i) => ({
      p: predictProbaBinary(model, row).p,
      y: y[i],
    }));
    probs.sort((a, b) => b.p - a.p);
    const positives = probs.filter((q) => q.y === 1).length;
    const negatives = probs.length - positives;

    const rocPoints: { fpr: number; tpr: number; threshold: number }[] = [];
    let tp = 0;
    let fp = 0;
    rocPoints.push({ fpr: 0, tpr: 0, threshold: 1 });
    for (const q of probs) {
      if (q.y === 1) tp++;
      else fp++;
      rocPoints.push({
        fpr: negatives === 0 ? 0 : fp / negatives,
        tpr: positives === 0 ? 0 : tp / positives,
        threshold: q.p,
      });
    }
    rocPoints.push({ fpr: 1, tpr: 1, threshold: 0 });

    let auc = 0;
    for (let i = 1; i < rocPoints.length; i++) {
      const dx = rocPoints[i].fpr - rocPoints[i - 1].fpr;
      const avg = (rocPoints[i].tpr + rocPoints[i - 1].tpr) / 2;
      auc += dx * avg;
    }
    result.rocPoints = rocPoints;
    result.auc = auc;
  } else {
    // Macro-averaged one-vs-rest AUC for multiclass
    let aucSum = 0;
    for (let c = 0; c < k; c++) {
      const probs = X.map((row, i) => ({
        p: predictProbaOvR(model, row).probs[c],
        y: y[i] === c ? 1 : 0,
      }));
      probs.sort((a, b) => b.p - a.p);
      const positives = probs.filter((q) => q.y === 1).length;
      const negatives = probs.length - positives;
      let tp = 0;
      let fp = 0;
      let prevFpr = 0;
      let prevTpr = 0;
      let auc = 0;
      for (const q of probs) {
        if (q.y === 1) tp++;
        else fp++;
        const fpr = negatives === 0 ? 0 : fp / negatives;
        const tpr = positives === 0 ? 0 : tp / positives;
        auc += (fpr - prevFpr) * ((tpr + prevTpr) / 2);
        prevFpr = fpr;
        prevTpr = tpr;
      }
      aucSum += auc;
    }
    result.auc = aucSum / k;
  }

  return result;
}
