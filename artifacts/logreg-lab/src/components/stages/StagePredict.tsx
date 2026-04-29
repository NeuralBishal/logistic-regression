import { useEffect, useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Cell } from "@/components/Notebook";
import { Math as Tex } from "@/components/Math";
import { useStore } from "@/lib/store";
import { transformSingle } from "@/lib/preprocess";
import { sigmoid, fmt, softmax, dot } from "@/lib/math-utils";
import { predictProbaOvR } from "@/lib/logreg";
import { Sparkles, Shuffle } from "lucide-react";
import { Badge } from "@/components/ui/badge";

export function StagePredict() {
  const { dataset, processed, training, preCfg, trainCfg } = useStore();
  const [values, setValues] = useState<Record<string, string>>({});

  // Initialise to the first row of the dataset for context
  useEffect(() => {
    if (!dataset || !preCfg) return;
    if (Object.keys(values).length > 0) return;
    const r = dataset.rows[0] ?? {};
    const next: Record<string, string> = {};
    for (const f of preCfg.featureColumns) {
      next[f] = r[f] === null || r[f] === undefined ? "" : String(r[f]);
    }
    setValues(next);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset, preCfg]);

  const result = useMemo(() => {
    if (!training || !processed || !preCfg || !dataset) return null;
    try {
      const xRaw: Record<string, string | number> = {};
      for (const f of preCfg.featureColumns) {
        const raw = values[f] ?? "";
        const col = dataset.columns.find((c) => c.name === f);
        if (col?.type === "numeric") {
          const n = Number(raw);
          xRaw[f] = Number.isFinite(n) ? n : 0;
        } else {
          xRaw[f] = raw;
        }
      }
      const x = transformSingle(
        xRaw,
        preCfg,
        processed.inputSchema,
        processed.scalingStats,
      );
      // Compute z and breakdown
      if (training.model.type === "binary") {
        const w = training.model.weights;
        const b = training.model.bias;
        const contributions = processed.featureNames.map((n, i) => ({
          feature: n,
          x: x[i],
          w: w[i],
          contribution: x[i] * w[i],
        }));
        const z = dot(x, w) + b;
        const p = sigmoid(z);
        const pred = p >= trainCfg.threshold ? 1 : 0;
        return {
          kind: "binary" as const,
          x,
          contributions,
          z,
          p,
          pred,
          bias: b,
        };
      } else {
        const { zs, probs } = predictProbaOvR(training.model, x);
        const sm = softmax(zs);
        let argmax = 0;
        for (let i = 1; i < probs.length; i++)
          if (probs[i] > probs[argmax]) argmax = i;
        return {
          kind: "multi" as const,
          x,
          probs,
          zs,
          sm,
          pred: argmax,
        };
      }
    } catch {
      return null;
    }
  }, [training, processed, preCfg, dataset, values, trainCfg.threshold]);

  if (!training || !processed || !preCfg || !dataset) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-muted-foreground">
          Train a model first.
        </CardContent>
      </Card>
    );
  }

  function loadRow(idx: number) {
    if (!dataset || !preCfg) return;
    const r = dataset.rows[idx] ?? {};
    const next: Record<string, string> = {};
    for (const f of preCfg.featureColumns) {
      next[f] = r[f] === null || r[f] === undefined ? "" : String(r[f]);
    }
    setValues(next);
  }

  return (
    <div className="space-y-6">
      <Cell kind="explain" title="Stage 7 — Make a prediction step by step">
        <p>
          Punch in feature values (or load an example) and watch the math run.
          Each value gets multiplied by its weight, the contributions add up to{" "}
          <Tex tex="z" />, the sigmoid squashes it into a probability, and the
          threshold turns that into a class.
        </p>
      </Cell>

      <div className="grid lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="text-base">Input features</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                className="flex-1"
                onClick={() =>
                  loadRow(Math.floor(Math.random() * dataset.rows.length))
                }
                data-testid="button-random-row"
              >
                <Shuffle className="w-4 h-4 mr-2" />
                Random row
              </Button>
              <Select
                onValueChange={(v) => loadRow(Number(v))}
              >
                <SelectTrigger className="w-32">
                  <SelectValue placeholder="Pick row" />
                </SelectTrigger>
                <SelectContent>
                  {dataset.rows.slice(0, 50).map((_, i) => (
                    <SelectItem key={i} value={String(i)}>
                      Row {i + 1}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2 max-h-[480px] overflow-y-auto pr-2">
              {preCfg.featureColumns.map((f) => {
                const col = dataset.columns.find((c) => c.name === f);
                if (col?.type === "categorical") {
                  const opts = Array.from(
                    new Set(
                      dataset.rows
                        .map((r) => r[f])
                        .filter((v) => v !== null && v !== undefined),
                    ),
                  ).map(String);
                  return (
                    <div key={f}>
                      <Label className="text-xs font-mono">{f}</Label>
                      <Select
                        value={values[f] ?? ""}
                        onValueChange={(v) =>
                          setValues({ ...values, [f]: v })
                        }
                      >
                        <SelectTrigger
                          className="h-8 text-sm"
                          data-testid={`input-${f}`}
                        >
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {opts.map((o) => (
                            <SelectItem key={o} value={o}>
                              {o}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  );
                }
                return (
                  <div key={f}>
                    <Label className="text-xs font-mono">{f}</Label>
                    <Input
                      type="number"
                      value={values[f] ?? ""}
                      onChange={(e) =>
                        setValues({ ...values, [f]: e.target.value })
                      }
                      className="h-8 text-sm font-mono"
                      data-testid={`input-${f}`}
                    />
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>

        <div className="lg:col-span-2 space-y-6">
          {result && result.kind === "binary" && (
            <>
              <Card>
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-accent" />
                    Step-by-step computation
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="overflow-x-auto rounded border max-h-72">
                    <table className="w-full text-xs font-mono">
                      <thead className="bg-muted/40 sticky top-0">
                        <tr>
                          <th className="text-left p-2">feature</th>
                          <th className="text-right p-2">x (scaled)</th>
                          <th className="text-right p-2">w</th>
                          <th className="text-right p-2">w · x</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.contributions
                          .slice()
                          .sort(
                            (a, b) =>
                              Math.abs(b.contribution) -
                              Math.abs(a.contribution),
                          )
                          .map((c) => (
                            <tr key={c.feature} className="border-t">
                              <td className="p-2 truncate max-w-[180px]">
                                {c.feature}
                              </td>
                              <td className="p-2 text-right tabular-nums">
                                {fmt(c.x, 3)}
                              </td>
                              <td className="p-2 text-right tabular-nums">
                                {fmt(c.w, 3)}
                              </td>
                              <td
                                className={`p-2 text-right tabular-nums font-bold ${
                                  c.contribution > 0
                                    ? "text-emerald-600 dark:text-emerald-400"
                                    : c.contribution < 0
                                      ? "text-rose-600 dark:text-rose-400"
                                      : ""
                                }`}
                              >
                                {fmt(c.contribution, 3)}
                              </td>
                            </tr>
                          ))}
                        <tr className="border-t bg-muted/30">
                          <td className="p-2 font-bold">+ bias</td>
                          <td colSpan={2}></td>
                          <td className="p-2 text-right tabular-nums font-bold">
                            {fmt(result.bias, 3)}
                          </td>
                        </tr>
                        <tr className="border-t bg-muted/50">
                          <td className="p-2 font-bold">= z</td>
                          <td colSpan={2}></td>
                          <td className="p-2 text-right tabular-nums font-bold text-primary">
                            {fmt(result.z, 3)}
                          </td>
                        </tr>
                      </tbody>
                    </table>
                  </div>

                  <div className="grid sm:grid-cols-3 gap-3">
                    <PredictKPI label="z (linear score)" value={fmt(result.z, 3)} />
                    <PredictKPI
                      label="σ(z) probability"
                      value={fmt(result.p, 4)}
                      accent
                    />
                    <PredictKPI
                      label={`Class @ threshold ${trainCfg.threshold}`}
                      value={processed.classNames[result.pred]}
                      accent
                    />
                  </div>

                  <div className="rounded-lg border bg-muted/30 p-4 text-sm space-y-1.5">
                    <div className="flex items-baseline gap-2">
                      <Tex tex="z = w \cdot x + b = " />
                      <span className="font-mono font-bold text-primary">
                        {fmt(result.z, 3)}
                      </span>
                    </div>
                    <div className="flex items-baseline gap-2">
                      <Tex
                        tex={`\\sigma(${fmt(result.z, 3)}) = \\frac{1}{1 + e^{${fmt(-result.z, 3)}}} = `}
                      />
                      <span className="font-mono font-bold text-accent">
                        {fmt(result.p, 4)}
                      </span>
                    </div>
                    <div className="flex items-baseline gap-2">
                      <span>
                        {fmt(result.p, 4)}{" "}
                        {result.p >= trainCfg.threshold ? "≥" : "<"}{" "}
                        {trainCfg.threshold} →{" "}
                      </span>
                      <Badge>
                        {processed.classNames[result.pred]}
                      </Badge>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <ProbabilityBar
                p={result.p}
                threshold={trainCfg.threshold}
                classNames={processed.classNames}
              />
            </>
          )}

          {result && result.kind === "multi" && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-accent" />
                  One-vs-Rest scores → softmax
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="overflow-x-auto rounded border">
                  <table className="w-full text-sm">
                    <thead className="bg-muted/40">
                      <tr>
                        <th className="text-left p-2">class</th>
                        <th className="text-right p-2">z</th>
                        <th className="text-right p-2">softmax</th>
                        <th className="text-left p-2"></th>
                      </tr>
                    </thead>
                    <tbody>
                      {processed.classNames.map((c, i) => (
                        <tr
                          key={c}
                          className={`border-t ${i === result.pred ? "bg-accent/10" : ""}`}
                        >
                          <td className="p-2 font-mono text-xs">{c}</td>
                          <td className="p-2 text-right font-mono text-xs tabular-nums">
                            {fmt(result.zs[i], 3)}
                          </td>
                          <td className="p-2 text-right font-mono text-xs tabular-nums font-bold">
                            {fmt(result.sm[i], 4)}
                          </td>
                          <td className="p-2 w-1/2">
                            <div className="h-3 bg-muted rounded overflow-hidden">
                              <div
                                className="h-full bg-primary"
                                style={{ width: `${result.sm[i] * 100}%` }}
                              />
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="text-sm">
                  Argmax → predicted class:{" "}
                  <Badge className="text-base">
                    {processed.classNames[result.pred]}
                  </Badge>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

function PredictKPI({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div className="rounded-lg border bg-card p-3">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div
        className={`text-xl font-semibold tabular-nums mt-1 ${accent ? "text-accent" : ""}`}
      >
        {value}
      </div>
    </div>
  );
}

function ProbabilityBar({
  p,
  threshold,
  classNames,
}: {
  p: number;
  threshold: number;
  classNames: string[];
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Probability gauge</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="relative h-12 bg-gradient-to-r from-rose-200 via-amber-100 to-emerald-200 dark:from-rose-950 dark:via-amber-950 dark:to-emerald-950 rounded-md overflow-visible">
          <div
            className="absolute top-0 bottom-0 w-px bg-foreground/60"
            style={{ left: `${threshold * 100}%` }}
          />
          <div
            className="absolute -top-1 -bottom-1 w-1 bg-primary rounded shadow-md"
            style={{ left: `calc(${p * 100}% - 2px)` }}
          />
          <div
            className="absolute -bottom-6 text-[10px] text-muted-foreground"
            style={{ left: `${threshold * 100}%`, transform: "translateX(-50%)" }}
          >
            threshold {threshold}
          </div>
          <div
            className="absolute -top-6 text-xs font-mono font-bold"
            style={{ left: `${p * 100}%`, transform: "translateX(-50%)" }}
          >
            {fmt(p, 3)}
          </div>
        </div>
        <div className="flex justify-between text-xs text-muted-foreground mt-8">
          <span>{classNames[0]} (0)</span>
          <span>{classNames[1]} (1)</span>
        </div>
      </CardContent>
    </Card>
  );
}
