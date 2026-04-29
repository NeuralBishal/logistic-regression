import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Cell } from "@/components/Notebook";
import { useStore } from "@/lib/store";
import { binCounts, correlation, fmt } from "@/lib/math-utils";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RTooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ZAxis,
  Cell as RechartsCell,
} from "recharts";
import { ArrowRight } from "lucide-react";

export function StageEDA() {
  const { dataset, processed, preCfg, setActiveStage } = useStore();
  const [chosen, setChosen] = useState<string | null>(null);

  const featureCol = chosen ?? processed?.featureNames[0] ?? "";
  const featureIdx = processed?.featureNames.indexOf(featureCol) ?? -1;

  const classCounts = useMemo(() => {
    if (!processed) return [];
    return processed.classNames.map((name, i) => ({
      name,
      count: processed.y.filter((v) => v === i).length,
    }));
  }, [processed]);

  const histAll = useMemo(() => {
    if (!processed || featureIdx < 0) return [];
    const featureValues = processed.featureMatrix.map(
      (row) => row[featureIdx],
    );
    return binCounts(featureValues, 20);
  }, [processed, featureIdx]);

  const histByClass = useMemo(() => {
    if (!processed || featureIdx < 0) return [];
    const byClass = processed.classNames.map((_, c) =>
      binCounts(
        processed.featureMatrix
          .filter((_, i) => processed.y[i] === c)
          .map((row) => row[featureIdx]),
        20,
      ),
    );
    return histAll.map((b, i) => {
      const out: Record<string, number | string> = { bin: b.bin };
      for (let c = 0; c < processed.classNames.length; c++) {
        out[processed.classNames[c]] = byClass[c][i]?.count ?? 0;
      }
      return out;
    });
  }, [processed, featureIdx, histAll]);

  const corrRows = useMemo(() => {
    if (!processed) return [];
    const out: { feature: string; r: number }[] = [];
    for (let j = 0; j < processed.featureNames.length; j++) {
      const col = processed.featureMatrix.map((row) => row[j]);
      const r = correlation(col, processed.y.map((v) => v));
      out.push({ feature: processed.featureNames[j], r });
    }
    return out.sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
  }, [processed]);

  const scatterData = useMemo(() => {
    if (!processed || featureIdx < 0) return [];
    return processed.featureMatrix.map((row, i) => ({
      x: row[featureIdx],
      y: processed.y[i] + (Math.random() - 0.5) * 0.2,
      cls: processed.y[i],
    }));
  }, [processed, featureIdx]);

  if (!dataset || !processed || !preCfg) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-muted-foreground">
          Apply preprocessing first.
        </CardContent>
      </Card>
    );
  }

  const colors = ["#7c5cff", "#22b8a5", "#f59e0b", "#ec4899", "#0ea5e9"];

  return (
    <div className="space-y-6">
      <Cell kind="explain" title="Stage 3 — Look before you leap">
        <p>
          Before training, eyeball the data. Imbalanced classes? Features that
          already separate the classes? Features that don't budge with the
          target? EDA tells you what to expect.
        </p>
      </Cell>

      <div className="grid lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Class distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={classCounts}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} />
                <RTooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--popover))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: 6,
                    fontSize: 12,
                  }}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {classCounts.map((_, i) => (
                    <RechartsCell key={i} fill={colors[i % colors.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="text-xs text-muted-foreground mt-2">
              {classCounts.length === 2 &&
                Math.abs(classCounts[0].count - classCounts[1].count) /
                  Math.max(classCounts[0].count, classCounts[1].count) >
                  0.3 &&
                "Classes look imbalanced — accuracy alone may be misleading. Check precision/recall on the evaluation page."}
              {classCounts.length === 2 &&
                Math.abs(classCounts[0].count - classCounts[1].count) /
                  Math.max(classCounts[0].count, classCounts[1].count) <=
                  0.3 &&
                "Classes are roughly balanced — accuracy will be a fair metric."}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">
              Correlation with target
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart
                data={corrRows.slice(0, 10)}
                layout="vertical"
                margin={{ left: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  type="number"
                  domain={[-1, 1]}
                  tick={{ fontSize: 11 }}
                />
                <YAxis
                  type="category"
                  dataKey="feature"
                  width={120}
                  tick={{ fontSize: 11 }}
                />
                <RTooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--popover))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: 6,
                    fontSize: 12,
                  }}
                  formatter={(v: number) => fmt(v, 3)}
                />
                <Bar dataKey="r" radius={[0, 4, 4, 0]}>
                  {corrRows.slice(0, 10).map((d, i) => (
                    <RechartsCell
                      key={i}
                      fill={
                        d.r >= 0
                          ? "hsl(var(--chart-1))"
                          : "hsl(var(--destructive))"
                      }
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <p className="text-xs text-muted-foreground mt-2">
              Top 10 by absolute correlation with the (encoded) target. Values
              near ±1 mean the feature moves with the target; near 0 means no
              linear relationship.
            </p>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-base">Feature explorer</CardTitle>
          <Select value={featureCol} onValueChange={setChosen}>
            <SelectTrigger className="w-72" data-testid="select-eda-feature">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {processed.featureNames
                .filter((n) => n !== "")
                .map((n) => (
                  <SelectItem key={n} value={n}>
                    {n}
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>
        </CardHeader>
        <CardContent className="grid lg:grid-cols-2 gap-6">
          <div>
            <h4 className="text-sm font-medium mb-2">
              Distribution by class
            </h4>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={histByClass}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="bin"
                  tick={{ fontSize: 10 }}
                  interval={Math.floor(histByClass.length / 6)}
                />
                <YAxis tick={{ fontSize: 11 }} />
                <RTooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--popover))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: 6,
                    fontSize: 12,
                  }}
                />
                {processed.classNames.map((c, i) => (
                  <Bar
                    key={c}
                    dataKey={c}
                    stackId="a"
                    fill={colors[i % colors.length]}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
            <p className="text-xs text-muted-foreground mt-2">
              When the colored bars sit in different ranges, this feature{" "}
              <em>helps separate</em> the classes — exactly what the model is
              looking for.
            </p>
          </div>

          <div>
            <h4 className="text-sm font-medium mb-2">
              Feature vs target (jittered)
            </h4>
            <ResponsiveContainer width="100%" height={250}>
              <ScatterChart>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="hsl(var(--border))"
                />
                <XAxis
                  type="number"
                  dataKey="x"
                  name={featureCol}
                  tick={{ fontSize: 11 }}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name="class"
                  tick={{ fontSize: 11 }}
                  domain={[-0.5, processed.classNames.length - 0.5]}
                  ticks={processed.classNames.map((_, i) => i)}
                  tickFormatter={(v: number) =>
                    processed.classNames[Math.round(v)] ?? ""
                  }
                />
                <ZAxis range={[20, 20]} />
                <RTooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  contentStyle={{
                    backgroundColor: "hsl(var(--popover))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: 6,
                    fontSize: 12,
                  }}
                  formatter={(v: number) => fmt(v, 3)}
                />
                <Scatter data={scatterData}>
                  {scatterData.map((d, i) => (
                    <RechartsCell
                      key={i}
                      fill={colors[d.cls % colors.length]}
                      fillOpacity={0.55}
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <div className="flex justify-end">
        <Button
          onClick={() => setActiveStage("theory")}
          data-testid="button-continue-eda"
        >
          Learn how logistic regression works
          <ArrowRight className="w-4 h-4 ml-2" />
        </Button>
      </div>
    </div>
  );
}
