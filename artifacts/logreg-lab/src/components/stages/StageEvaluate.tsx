import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Cell } from "@/components/Notebook";
import { useStore } from "@/lib/store";
import { fmt } from "@/lib/math-utils";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RTooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  AreaChart,
} from "recharts";
import { ArrowRight } from "lucide-react";

export function StageEvaluate() {
  const { training, setActiveStage } = useStore();

  if (!training) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-muted-foreground">
          Train the model first.
        </CardContent>
      </Card>
    );
  }

  const t = training.testEval;
  const tr = training.trainEval;

  return (
    <div className="space-y-6">
      <Cell kind="explain" title="Stage 6 — How well did it do?">
        <p>
          Accuracy alone can lie — especially with imbalanced classes. Look at
          the full picture: confusion matrix shows what kinds of mistakes are
          happening, precision/recall reveal trade-offs, ROC/AUC summarises
          performance across every threshold.
        </p>
      </Cell>

      <div className="grid md:grid-cols-4 gap-4">
        <KPI label="Accuracy (test)" value={t.accuracy} />
        <KPI label="Macro F1 (test)" value={t.macroF1} />
        <KPI label="AUC (test)" value={t.auc ?? 0} />
        <KPI
          label="Train accuracy"
          value={tr.accuracy}
          subtitle={
            tr.accuracy - t.accuracy > 0.1
              ? "⚠ likely overfitting"
              : "matches test ✓"
          }
        />
      </div>

      <div className="grid lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Confusion matrix (test)</CardTitle>
          </CardHeader>
          <CardContent>
            <ConfusionMatrixView matrix={t.confusion.matrix} classNames={t.confusion.classNames} />
            <p className="text-xs text-muted-foreground mt-3">
              Rows = actual class, columns = predicted class. Diagonal = correct;
              off-diagonal = mistakes.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Per-class metrics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="overflow-x-auto rounded border">
              <table className="w-full text-sm">
                <thead className="bg-muted/40">
                  <tr>
                    <th className="text-left p-2 font-medium">Class</th>
                    <th className="text-right p-2 font-medium">Precision</th>
                    <th className="text-right p-2 font-medium">Recall</th>
                    <th className="text-right p-2 font-medium">F1</th>
                    <th className="text-right p-2 font-medium">Support</th>
                  </tr>
                </thead>
                <tbody>
                  {t.perClass.map((c) => (
                    <tr key={c.className} className="border-t">
                      <td className="p-2 font-mono text-xs">{c.className}</td>
                      <td className="p-2 text-right tabular-nums font-mono text-xs">
                        {fmt(c.precision, 3)}
                      </td>
                      <td className="p-2 text-right tabular-nums font-mono text-xs">
                        {fmt(c.recall, 3)}
                      </td>
                      <td className="p-2 text-right tabular-nums font-mono text-xs">
                        {fmt(c.f1, 3)}
                      </td>
                      <td className="p-2 text-right tabular-nums font-mono text-xs">
                        {c.support}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-3 text-xs text-muted-foreground space-y-1">
              <p>
                <strong>Precision</strong>: of the times we said class X, how
                often were we right?
              </p>
              <p>
                <strong>Recall</strong>: of all real X's, how many did we catch?
              </p>
              <p>
                <strong>F1</strong>: harmonic mean — penalises imbalance between
                the two.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {t.rocPoints && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">
              ROC curve — AUC = {fmt(t.auc ?? 0, 4)}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart
                data={t.rocPoints}
                margin={{ top: 10, right: 20, bottom: 20, left: 10 }}
              >
                <defs>
                  <linearGradient id="aucG" x1="0" y1="0" x2="0" y2="1">
                    <stop
                      offset="0%"
                      stopColor="hsl(var(--chart-1))"
                      stopOpacity={0.4}
                    />
                    <stop
                      offset="100%"
                      stopColor="hsl(var(--chart-1))"
                      stopOpacity={0}
                    />
                  </linearGradient>
                </defs>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="hsl(var(--border))"
                />
                <XAxis
                  type="number"
                  dataKey="fpr"
                  domain={[0, 1]}
                  tick={{ fontSize: 11 }}
                  label={{
                    value: "False positive rate",
                    position: "insideBottom",
                    offset: -5,
                    fontSize: 11,
                  }}
                />
                <YAxis
                  domain={[0, 1]}
                  tick={{ fontSize: 11 }}
                  label={{
                    value: "True positive rate",
                    angle: -90,
                    position: "insideLeft",
                    fontSize: 11,
                  }}
                />
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
                  segment={[
                    { x: 0, y: 0 },
                    { x: 1, y: 1 },
                  ]}
                  stroke="hsl(var(--muted-foreground))"
                  strokeDasharray="4 4"
                />
                <Area
                  type="monotone"
                  dataKey="tpr"
                  stroke="hsl(var(--chart-1))"
                  strokeWidth={2}
                  fill="url(#aucG)"
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
            <p className="text-xs text-muted-foreground mt-2">
              The curve sweeps every possible threshold. AUC = 1 is perfect, 0.5
              is random guessing. The dashed diagonal is what a coin flip would
              do.
            </p>
          </CardContent>
        </Card>
      )}

      {training.cv && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">
              {training.cv.k}-fold cross-validation
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex items-baseline gap-3 mb-3">
              <div className="text-3xl font-semibold tabular-nums">
                {fmt(training.cv.mean, 4)}
              </div>
              <div className="text-sm text-muted-foreground">
                ± {fmt(training.cv.std, 4)} accuracy across folds
              </div>
            </div>
            <ResponsiveContainer width="100%" height={180}>
              <LineChart
                data={training.cv.foldScores.map((s, i) => ({
                  fold: i + 1,
                  acc: s,
                }))}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="hsl(var(--border))"
                />
                <XAxis dataKey="fold" tick={{ fontSize: 11 }} />
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
                  y={training.cv.mean}
                  stroke="hsl(var(--accent))"
                  strokeDasharray="4 4"
                />
                <Line
                  type="monotone"
                  dataKey="acc"
                  stroke="hsl(var(--chart-1))"
                  strokeWidth={2}
                  dot={{ r: 4 }}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
            <p className="text-xs text-muted-foreground mt-2">
              Cross-validation gives you a more honest estimate of the model's
              generalization than a single train/test split. Tight fold-to-fold
              variance means stable performance.
            </p>
          </CardContent>
        </Card>
      )}

      <div className="flex justify-end">
        <Button
          onClick={() => setActiveStage("predict")}
          data-testid="button-continue-eval"
        >
          Try a prediction
          <ArrowRight className="w-4 h-4 ml-2" />
        </Button>
      </div>
    </div>
  );
}

function KPI({
  label,
  value,
  subtitle,
}: {
  label: string;
  value: number;
  subtitle?: string;
}) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="text-[11px] uppercase tracking-wider text-muted-foreground">
          {label}
        </div>
        <div className="text-3xl font-semibold tabular-nums mt-1">
          {fmt(value, 4)}
        </div>
        {subtitle && (
          <div className="text-[11px] text-muted-foreground mt-1">
            {subtitle}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ConfusionMatrixView({
  matrix,
  classNames,
}: {
  matrix: number[][];
  classNames: string[];
}) {
  const max = Math.max(...matrix.flat(), 1);
  return (
    <div className="overflow-x-auto">
      <table className="border-separate border-spacing-1">
        <thead>
          <tr>
            <th className="p-1"></th>
            <th
              colSpan={classNames.length}
              className="text-xs text-center text-muted-foreground pb-1 font-medium"
            >
              predicted
            </th>
          </tr>
          <tr>
            <th></th>
            {classNames.map((c) => (
              <th
                key={c}
                className="text-xs font-mono p-2 text-muted-foreground"
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th className="text-xs font-mono px-2 text-muted-foreground text-right whitespace-nowrap">
                {classNames[i]}
              </th>
              {row.map((v, j) => {
                const intensity = v / max;
                const isDiag = i === j;
                const bg = isDiag
                  ? `hsl(160 60% ${95 - intensity * 50}%)`
                  : `hsl(0 70% ${97 - intensity * 40}%)`;
                return (
                  <td
                    key={j}
                    className="p-3 min-w-[80px] text-center font-mono text-sm rounded border"
                    style={{
                      backgroundColor: bg,
                      color: intensity > 0.6 ? "white" : "hsl(230 25% 14%)",
                    }}
                  >
                    {v}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
