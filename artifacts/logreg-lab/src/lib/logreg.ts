import { sigmoid, softmax, dot, shuffleIndices, clip } from "./math-utils";

export interface BinaryLogRegModel {
  type: "binary";
  weights: number[]; // length d
  bias: number;
  classNames: [string, string];
  /** Loss per epoch — for the training-curve chart. */
  lossHistory: number[];
}

export interface OvRLogRegModel {
  type: "ovr";
  /** One binary model per class. */
  models: { weights: number[]; bias: number; lossHistory: number[] }[];
  classNames: string[];
}

export type LogRegModel = BinaryLogRegModel | OvRLogRegModel;

export interface TrainConfig {
  learningRate: number;
  epochs: number;
  l2: number;
  threshold: number; // for binary classification
}

export const DEFAULT_TRAIN_CONFIG: TrainConfig = {
  learningRate: 0.1,
  epochs: 200,
  l2: 0.01,
  threshold: 0.5,
};

function trainBinary(
  X: number[][],
  y: number[],
  cfg: TrainConfig,
): { weights: number[]; bias: number; lossHistory: number[] } {
  const n = X.length;
  const d = X[0]?.length ?? 0;
  const weights = new Array(d).fill(0);
  let bias = 0;
  const lossHistory: number[] = [];

  for (let epoch = 0; epoch < cfg.epochs; epoch++) {
    const dW = new Array(d).fill(0);
    let dB = 0;
    let loss = 0;

    for (let i = 0; i < n; i++) {
      const z = dot(X[i], weights) + bias;
      const p = sigmoid(z);
      const err = p - y[i];
      for (let j = 0; j < d; j++) dW[j] += err * X[i][j];
      dB += err;

      const pc = clip(p, 1e-12, 1 - 1e-12);
      loss += -(y[i] * Math.log(pc) + (1 - y[i]) * Math.log(1 - pc));
    }

    // L2 regularization gradient
    for (let j = 0; j < d; j++) {
      dW[j] = dW[j] / n + cfg.l2 * weights[j];
      weights[j] -= cfg.learningRate * dW[j];
    }
    bias -= cfg.learningRate * (dB / n);

    let reg = 0;
    for (let j = 0; j < d; j++) reg += weights[j] * weights[j];
    loss = loss / n + 0.5 * cfg.l2 * reg;
    lossHistory.push(loss);
  }

  return { weights, bias, lossHistory };
}

export function trainModel(
  X: number[][],
  y: number[],
  classNames: string[],
  cfg: TrainConfig,
): LogRegModel {
  if (classNames.length === 2) {
    const { weights, bias, lossHistory } = trainBinary(X, y, cfg);
    return {
      type: "binary",
      weights,
      bias,
      classNames: [classNames[0], classNames[1]],
      lossHistory,
    };
  }

  // One-vs-Rest
  const models = classNames.map((_, k) => {
    const yk = y.map((v) => (v === k ? 1 : 0));
    return trainBinary(X, yk, cfg);
  });
  return { type: "ovr", models, classNames };
}

export function predictProbaBinary(
  model: BinaryLogRegModel,
  x: number[],
): { z: number; p: number } {
  const z = dot(x, model.weights) + model.bias;
  return { z, p: sigmoid(z) };
}

export function predictProbaOvR(
  model: OvRLogRegModel,
  x: number[],
): { zs: number[]; rawSigmoids: number[]; probs: number[] } {
  const zs = model.models.map((m) => dot(x, m.weights) + m.bias);
  const rawSigmoids = zs.map((z) => sigmoid(z));
  const probs = softmax(zs);
  return { zs, rawSigmoids, probs };
}

export function predictClass(
  model: LogRegModel,
  x: number[],
  threshold: number,
): { label: number; name: string } {
  if (model.type === "binary") {
    const { p } = predictProbaBinary(model, x);
    const label = p >= threshold ? 1 : 0;
    return { label, name: model.classNames[label] };
  }
  const { probs } = predictProbaOvR(model, x);
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  return { label: best, name: model.classNames[best] };
}

export function trainTestSplit(
  X: number[][],
  y: number[],
  testSize: number,
  seed = 42,
): { Xtrain: number[][]; ytrain: number[]; Xtest: number[][]; ytest: number[] } {
  const n = X.length;
  const idx = shuffleIndices(n, seed);
  const nTest = Math.max(1, Math.round(n * testSize));
  const testIdx = idx.slice(0, nTest);
  const trainIdx = idx.slice(nTest);
  return {
    Xtrain: trainIdx.map((i) => X[i]),
    ytrain: trainIdx.map((i) => y[i]),
    Xtest: testIdx.map((i) => X[i]),
    ytest: testIdx.map((i) => y[i]),
  };
}

export function kFoldCrossValidate(
  X: number[][],
  y: number[],
  classNames: string[],
  k: number,
  cfg: TrainConfig,
): { foldScores: number[]; mean: number; std: number } {
  const n = X.length;
  const idx = shuffleIndices(n, 7);
  const foldSize = Math.floor(n / k);
  const foldScores: number[] = [];

  for (let f = 0; f < k; f++) {
    const valIdx = idx.slice(f * foldSize, (f + 1) * foldSize);
    const trainIdx = idx.filter((_, i) => i < f * foldSize || i >= (f + 1) * foldSize);
    const Xtr = trainIdx.map((i) => X[i]);
    const ytr = trainIdx.map((i) => y[i]);
    const Xva = valIdx.map((i) => X[i]);
    const yva = valIdx.map((i) => y[i]);
    const m = trainModel(Xtr, ytr, classNames, cfg);
    let correct = 0;
    for (let i = 0; i < Xva.length; i++) {
      const { label } = predictClass(m, Xva[i], cfg.threshold);
      if (label === yva[i]) correct++;
    }
    foldScores.push(Xva.length > 0 ? correct / Xva.length : 0);
  }

  const m = foldScores.reduce((s, v) => s + v, 0) / foldScores.length;
  const v =
    foldScores.reduce((s, x) => s + (x - m) * (x - m), 0) / foldScores.length;
  return { foldScores, mean: m, std: Math.sqrt(v) };
}
