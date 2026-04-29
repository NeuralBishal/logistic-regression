import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Cell } from "@/components/Notebook";
import { Math as Tex } from "@/components/Math";
import { Tip } from "@/components/Tip";
import { useStore } from "@/lib/store";
import {
  trainModel,
  trainTestSplit,
  kFoldCrossValidate,
} from "@/lib/logreg";
import { evaluate } from "@/lib/metrics";
import { fmt, sigmoid } from "@/lib/math-utils";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RTooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea,
  Scatter,
  ComposedChart,
  Legend,
  ZAxis,
  Cell as RCell,
} from "recharts";
import { Play, ArrowRight } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export function StageTrain() {
  const {
    processed,
    trainCfg,
    setTrainCfg,
    testSize,
    setTestSize,
    useCV,
    setUseCV,
    cvK,
    setCvK,
    training,
    setTraining,
    setActiveStage,
  } = useStore();
  const [busy, setBusy] = useState(false);

  // Pick two features for the decision-boundary chart
  const allFeatures = processed?.featureNames ?? [];
  const [feat1, setFeat1] = useState<string>(allFeatures[0] ?? "");
  const [feat2, setFeat2] = useState<string>(allFeatures[1] ?? "");

  if (!processed) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-muted-foreground">
          Apply preprocessing first.
        </CardContent>
      </Card>
    );
  }

  async function handleTrain() {
    if (!processed) return;
    setBusy(true);
    // Yield a tick so the button shows the busy state
    await new Promise((r) => setTimeout(r, 30));
    try {
      const { Xtrain, ytrain, Xtest, ytest } = trainTestSplit(
        processed.featureMatrix,
        processed.y,
        testSize,
      );
      const model = trainModel(Xtrain, ytrain, processed.classNames, trainCfg);
      const trainEval = evaluate(model, Xtrain, ytrain, trainCfg.threshold);
      const testEval = evaluate(model, Xtest, ytest, trainCfg.threshold);
      let cv = undefined;
      if (useCV) {
        const r = kFoldCrossValidate(
          processed.featureMatrix,
          processed.y,
          processed.classNames,
          cvK,
          trainCfg,
        );
        cv = { ...r, k: cvK };
      }
      setTraining({
        model,
        Xtrain,
        ytrain,
        Xtest,
        ytest,
        trainEval,
        testEval,
        cv,
      });
    } finally {
      setBusy(false);
    }
  }

  // Loss curve data
  const lossData = useMemo(() => {
    if (!training) return [];
    if (training.model.type === "binary") {
      return training.model.lossHistory.map((l, i) => ({ epoch: i, loss: l }));
    }
    // average across OvR models
    const len = training.model.models[0]?.lossHistory.length ?? 0;
    const out: { epoch: number; loss: number }[] = [];
    for (let i = 0; i < len; i++) {
      let s = 0;
      for (const m of training.model.models) s += m.lossHistory[i];
      out.push({ epoch: i, loss: s / training.model.models.length });
    }
    return out;
  }, [training]);

  // 2D decision boundary
  const idx1 = allFeatures.indexOf(feat1);
  const idx2 = allFeatures.indexOf(feat2);
  const boundary = useMemo(() => {
    if (!training || idx1 < 0 || idx2 < 0 || idx1 === idx2) return null;
    if (training.model.type !== "binary") return null;
    const w1 = training.model.weights[idx1];
    const w2 = training.model.weights[idx2];
    // Use mean of other features (≈ 0 after standard scaling) so the
    // visualised boundary uses the trained bias plus the other features' contribution.
    let baseZ = training.model.bias;
    for (let j = 0; j < training.model.weights.length; j++) {
      if (j === idx1 || j === idx2) continue;
      const m =
        processed.featureMatrix.reduce((s, r) => s + r[j], 0) /
        processed.featureMatrix.length;
      baseZ += training.model.weights[j] * m;
    }
    // Range
    const xs = processed.featureMatrix.map((r) => r[idx1]);
    const ys = processed.featureMatrix.map((r) => r[idx2]);
    const xmin = Math.min(...xs);
    const xmax = Math.max(...xs);
    const points: { x: number; y: number }[] = [];
    if (Math.abs(w2) > 1e-9) {
      for (let xv = xmin; xv <= xmax; xv += (xmax - xmin) / 50) {
        const yv = -(baseZ + w1 * xv) / w2;
        points.push({ x: xv, y: yv });
      }
    } else if (Math.abs(w1) > 1e-9) {
      const xv = -baseZ / w1;
      const ymin = Math.min(...ys);
      const ymax = Math.max(...ys);
      points.push({ x: xv, y: ymin });
      points.push({ x: xv, y: ymax });
    }
    return points;
  }, [training, processed, idx1, idx2]);

  const scatterPoints = useMemo(() => {
    if (!training || idx1 < 0 || idx2 < 0) return [];
    return processed.featureMatrix.map((r, i) => ({
      x: r[idx1],
      y: r[idx2],
      cls: processed.y[i],
    }));
  }, [processed, training, idx1, idx2]);

  // Probability curve along feat1 holding others at mean
  const probCurve = useMemo(() => {
    if (!training || training.model.type !== "binary" || idx1 < 0) return [];
    const mean = new Array(training.model.weights.length).fill(0).map((_, j) => {
      return (
        processed.featureMatrix.reduce((s, r) => s + r[j], 0) /
        processed.featureMatrix.length
      );
    });
    const xs = processed.featureMatrix.map((r) => r[idx1]);
    const xmin = Math.min(...xs);
    const xmax = Math.max(...xs);
    const out: { x: number; p: number }[] = [];
    for (let v = xmin; v <= xmax; v += (xmax - xmin) / 60) {
      const x = mean.slice();
      x[idx1] = v;
      let z = training.model.bias;
      for (let j = 0; j < x.length; j++) z += training.model.weights[j] * x[j];
      out.push({ x: v, p: sigmoid(z) });
    }
    return out;
  }, [processed, training, idx1]);

  const colors = ["#7c5cff", "#22b8a5", "#f59e0b", "#ec4899", "#0ea5e9"];

  return (
    <div className="space-y-6">
      <Cell kind="explain" title="Stage 5 — Train the model">
        <p>
          You set the knobs; gradient descent fits the weights. With small
          datasets training is instant — the loss curve shows the model getting
          better epoch by epoch.
        </p>
      </Cell>

      <div className="grid lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-5">
            <div>
              <div className="flex items-center justify-between text-sm">
                <Label>Test size</Label>
                <span className="font-mono text-xs">
                  {Math.round(testSize * 100)}%
                </span>
              </div>
              <Slider
                value={[testSize]}
                onValueChange={(v) => setTestSize(v[0])}
                min={0.1}
                max={0.5}
                step={0.05}
                data-testid="slider-test-size"
              />
              <p className="text-[11px] text-muted-foreground mt-1">
                Held-out rows the model never sees during training.
              </p>
            </div>

            <div>
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-1.5">
                  <Label>Learning rate</Label>
                  <Tip>
                    How big each gradient step is. Too large = bouncy loss; too
                    small = slow training.
                  </Tip>
                </div>
                <span className="font-mono text-xs">
                  {trainCfg.learningRate.toFixed(2)}
                </span>
              </div>
              <Slider
                value={[trainCfg.learningRate]}
                onValueChange={(v) =>
                  setTrainCfg({ ...trainCfg, learningRate: v[0] })
                }
                min={0.01}
                max={1}
                step={0.01}
                data-testid="slider-lr"
              />
            </div>

            <div>
              <div className="flex items-center justify-between text-sm">
                <Label>Epochs</Label>
                <span className="font-mono text-xs">{trainCfg.epochs}</span>
              </div>
              <Slider
                value={[trainCfg.epochs]}
                onValueChange={(v) =>
                  setTrainCfg({ ...trainCfg, epochs: v[0] })
                }
                min={20}
                max={1000}
                step={20}
                data-testid="slider-epochs"
              />
            </div>

            <div>
              <div className="flex items-center justify-between text-sm">
                <div className="flex items-center gap-1.5">
                  <Label>L2 regularization</Label>
                  <Tip>
                    Penalty on big weights. Helps generalization, prevents
                    overfitting.
                  </Tip>
                </div>
                <span className="font-mono text-xs">
                  {trainCfg.l2.toFixed(3)}
                </span>
              </div>
              <Slider
                value={[trainCfg.l2]}
                onValueChange={(v) => setTrainCfg({ ...trainCfg, l2: v[0] })}
                min={0}
                max={0.5}
                step={0.005}
                data-testid="slider-l2"
              />
            </div>

            <div>
              <div className="flex items-center justify-between text-sm">
                <Label>Decision threshold</Label>
                <span className="font-mono text-xs">
                  {trainCfg.threshold.toFixed(2)}
                </span>
              </div>
              <Slider
                value={[trainCfg.threshold]}
                onValueChange={(v) =>
                  setTrainCfg({ ...trainCfg, threshold: v[0] })
                }
                min={0.05}
                max={0.95}
                step={0.05}
                data-testid="slider-threshold"
              />
              <p className="text-[11px] text-muted-foreground mt-1">
                Probability above this → class 1.
              </p>
            </div>

            <div className="space-y-3 pt-2 border-t">
              <div className="flex items-center justify-between">
                <Label className="text-sm">k-fold cross-validation</Label>
                <Switch
                  checked={useCV}
                  onCheckedChange={setUseCV}
                  data-testid="switch-cv"
                />
              </div>
              {useCV && (
                <div>
                  <div className="flex items-center justify-between text-sm">
                    <Label>K (folds)</Label>
                    <span className="font-mono text-xs">{cvK}</span>
                  </div>
                  <Slider
                    value={[cvK]}
                    onValueChange={(v) => setCvK(Math.round(v[0]))}
                    min={3}
                    max={10}
                    step={1}
                    data-testid="slider-k"
                  />
                </div>
              )}
            </div>

            <Button
              className="w-full"
              onClick={handleTrain}
              disabled={busy}
              data-testid="button-train"
            >
              <Play className="w-4 h-4 mr-2" />
              {busy ? "Training..." : "Train model"}
            </Button>
          </CardContent>
        </Card>

        <div className="lg:col-span-2 space-y-6">
          {!training && (
            <Card>
              <CardContent className="py-12 text-center text-muted-foreground">
                Hit "Train model" to fit logistic regression on{" "}
                {processed.featureMatrix.length} samples.
              </CardContent>
            </Card>
          )}

          {training && (
            <>
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Training loss</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={240}>
                    <LineChart data={lossData}>
                      <CartesianGrid
                        strokeDasharray="3 3"
                        stroke="hsl(var(--border))"
                      />
                      <XAxis
                        dataKey="epoch"
                        tick={{ fontSize: 11 }}
                        label={{
                          value: "epoch",
                          position: "insideBottom",
                          offset: -5,
                          fontSize: 11,
                        }}
                      />
                      <YAxis tick={{ fontSize: 11 }} />
                      <RTooltip
                        contentStyle={{
                          backgroundColor: "hsl(var(--popover))",
                          border: "1px solid hsl(var(--border))",
                          borderRadius: 6,
                          fontSize: 12,
                        }}
                        formatter={(v: number) => fmt(v, 4)}
                      />
                      <Line
                        type="monotone"
                        dataKey="loss"
                        stroke="hsl(var(--chart-1))"
                        strokeWidth={2}
                        dot={false}
                        isAnimationActive={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                  <p className="text-xs text-muted-foreground mt-2">
                    Final loss:{" "}
                    <span className="font-mono">
                      {fmt(lossData[lossData.length - 1]?.loss ?? 0, 4)}
                    </span>{" "}
                    — if it's still trending downward, increase epochs.
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-base">
                    Learned parameters
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <WeightsView
                    featureNames={processed.featureNames}
                    model={training.model}
                  />
                </CardContent>
              </Card>

              {training.model.type === "binary" && (
                <Card>
                  <CardHeader className="flex flex-row items-center justify-between">
                    <CardTitle className="text-base">
                      Decision boundary (2 features)
                    </CardTitle>
                    <div className="flex gap-2">
                      <Select value={feat1} onValueChange={setFeat1}>
                        <SelectTrigger className="w-32 h-8 text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {allFeatures.map((n) => (
                            <SelectItem key={n} value={n}>
                              {n}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <Select value={feat2} onValueChange={setFeat2}>
                        <SelectTrigger className="w-32 h-8 text-xs">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {allFeatures.map((n) => (
                            <SelectItem key={n} value={n}>
                              {n}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <ComposedChart>
                        <CartesianGrid
                          strokeDasharray="3 3"
                          stroke="hsl(var(--border))"
                        />
                        <XAxis
                          type="number"
                          dataKey="x"
                          name={feat1}
                          tick={{ fontSize: 11 }}
                          label={{
                            value: feat1,
                            position: "insideBottom",
                            offset: -5,
                            fontSize: 11,
                          }}
                        />
                        <YAxis
                          type="number"
                          dataKey="y"
                          name={feat2}
                          tick={{ fontSize: 11 }}
                          label={{
                            value: feat2,
                            angle: -90,
                            position: "insideLeft",
                            fontSize: 11,
                          }}
                        />
                        <ZAxis range={[40, 40]} />
                        <RTooltip
                          cursor={{ strokeDasharray: "3 3" }}
                          contentStyle={{
                            backgroundColor: "hsl(var(--popover))",
                            border: "1px solid hsl(var(--border))",
                            borderRadius: 6,
                            fontSize: 12,
                          }}
                        />
                        <Legend wrapperStyle={{ fontSize: 11 }} />
                        <Scatter
                          name={processed.classNames[0]}
                          data={scatterPoints.filter((p) => p.cls === 0)}
                          fill={colors[0]}
                          fillOpacity={0.55}
                        />
                        <Scatter
                          name={processed.classNames[1]}
                          data={scatterPoints.filter((p) => p.cls === 1)}
                          fill={colors[1]}
                          fillOpacity={0.55}
                        />
                        {boundary && boundary.length > 0 && (
                          <Line
                            data={boundary}
                            type="linear"
                            dataKey="y"
                            stroke="hsl(var(--foreground))"
                            strokeWidth={2}
                            strokeDasharray="6 4"
                            dot={false}
                            name="decision boundary"
                            isAnimationActive={false}
                          />
                        )}
                      </ComposedChart>
                    </ResponsiveContainer>
                    <p className="text-xs text-muted-foreground mt-2">
                      The dashed line is where{" "}
                      <Tex tex="\hat{p} = 0.5" />. Points are coloured by their
                      true class. Use the dropdowns to inspect different feature
                      pairs.
                    </p>
                  </CardContent>
                </Card>
              )}

              {training.model.type === "binary" && probCurve.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">
                      Probability curve along {feat1}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={220}>
                      <LineChart data={probCurve}>
                        <CartesianGrid
                          strokeDasharray="3 3"
                          stroke="hsl(var(--border))"
                        />
                        <XAxis
                          dataKey="x"
                          tick={{ fontSize: 11 }}
                          tickFormatter={(v: number) => fmt(v, 2)}
                        />
                        <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} />
                        <RTooltip
                          contentStyle={{
                            backgroundColor: "hsl(var(--popover))",
                            border: "1px solid hsl(var(--border))",
                            borderRadius: 6,
                            fontSize: 12,
                          }}
                          formatter={(v: number) => fmt(v, 4)}
                        />
                        <ReferenceLine
                          y={trainCfg.threshold}
                          stroke="hsl(var(--muted-foreground))"
                          strokeDasharray="4 4"
                        />
                        <ReferenceArea
                          y1={trainCfg.threshold}
                          y2={1}
                          fill={colors[1]}
                          fillOpacity={0.08}
                        />
                        <Line
                          type="monotone"
                          dataKey="p"
                          stroke="hsl(var(--chart-1))"
                          strokeWidth={2}
                          dot={false}
                          isAnimationActive={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                    <p className="text-xs text-muted-foreground mt-2">
                      Hold all other features at their average and slide{" "}
                      {feat1} — see how the predicted probability changes. Above
                      the dashed threshold, prediction = "{processed.classNames[1]}".
                    </p>
                  </CardContent>
                </Card>
              )}

              <div className="flex justify-end gap-2">
                <Button
                  onClick={() => setActiveStage("evaluate")}
                  data-testid="button-continue-train"
                >
                  See evaluation metrics
                  <ArrowRight className="w-4 h-4 ml-2" />
                </Button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function WeightsView({
  featureNames,
  model,
}: {
  featureNames: string[];
  model: NonNullable<ReturnType<typeof useStore>["training"]>["model"];
}) {
  if (model.type === "binary") {
    const rows = featureNames
      .map((n, i) => ({ name: n, w: model.weights[i] }))
      .sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
    const maxAbs = Math.max(...rows.map((r) => Math.abs(r.w)), 1e-6);
    return (
      <div className="space-y-3">
        <div className="text-xs text-muted-foreground">
          bias <span className="font-mono">b = {fmt(model.bias, 4)}</span>
        </div>
        <div className="max-h-72 overflow-y-auto pr-2 space-y-1">
          {rows.map((r) => (
            <div key={r.name} className="flex items-center gap-2 text-xs">
              <span className="font-mono w-44 truncate" title={r.name}>
                {r.name}
              </span>
              <div className="flex-1 h-4 bg-muted rounded relative overflow-hidden">
                <div
                  className="absolute inset-y-0 bg-primary/60"
                  style={{
                    left: r.w >= 0 ? "50%" : `${50 - (Math.abs(r.w) / maxAbs) * 50}%`,
                    width: `${(Math.abs(r.w) / maxAbs) * 50}%`,
                  }}
                />
                <div className="absolute inset-y-0 left-1/2 w-px bg-border" />
              </div>
              <span className="font-mono w-20 text-right tabular-nums">
                {fmt(r.w, 4)}
              </span>
            </div>
          ))}
        </div>
      </div>
    );
  }
  return (
    <div className="space-y-3 text-xs">
      <p className="text-muted-foreground">
        One-vs-Rest: {model.classNames.length} binary classifiers, one per
        class.
      </p>
      <div className="grid sm:grid-cols-2 gap-3 max-h-80 overflow-y-auto">
        {model.models.map((m, k) => (
          <div key={k} className="rounded border p-2 bg-muted/30">
            <div className="font-medium mb-1">
              "{model.classNames[k]}" vs rest
            </div>
            <div className="font-mono">
              bias <span>{fmt(m.bias, 3)}</span>
            </div>
            <div className="space-y-0.5 mt-1 max-h-32 overflow-y-auto pr-1">
              {featureNames.map((n, i) => (
                <div
                  key={n}
                  className="flex justify-between font-mono text-[10px]"
                >
                  <span className="truncate">{n}</span>
                  <span>{fmt(m.weights[i], 3)}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
