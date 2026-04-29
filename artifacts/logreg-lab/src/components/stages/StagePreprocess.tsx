import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ArrowRight, Wand2, RotateCcw } from "lucide-react";
import { useStore } from "@/lib/store";
import { preprocess } from "@/lib/preprocess";
import { Cell, CodeBlock } from "@/components/Notebook";
import { Math as Tex } from "@/components/Math";
import { Tip } from "@/components/Tip";
import { fmt } from "@/lib/math-utils";
import { Checkbox } from "@/components/ui/checkbox";

export function StagePreprocess() {
  const {
    dataset,
    preCfg,
    setPreCfg,
    setProcessed,
    processed,
    setActiveStage,
    setTraining,
  } = useStore();
  const computed = useMemo(() => {
    if (!dataset || !preCfg) return { result: null, error: null as string | null };
    try {
      return { result: preprocess(dataset, preCfg), error: null };
    } catch (e) {
      return {
        result: null,
        error: e instanceof Error ? e.message : String(e),
      };
    }
  }, [dataset, preCfg]);
  const result = computed.result;
  const error = computed.error;

  if (!dataset || !preCfg) {
    return (
      <Card>
        <CardContent className="py-8 text-center text-muted-foreground">
          Load a dataset first.
        </CardContent>
      </Card>
    );
  }

  const featureColumns = dataset.columns.filter(
    (c) => c.name !== preCfg.targetColumn,
  );

  function toggleFeature(name: string) {
    if (!preCfg) return;
    const has = preCfg.featureColumns.includes(name);
    setPreCfg({
      ...preCfg,
      featureColumns: has
        ? preCfg.featureColumns.filter((f) => f !== name)
        : [...preCfg.featureColumns, name],
    });
  }

  function applyAndContinue() {
    if (!result) return;
    setProcessed(result.processed);
    setTraining(null);
    setActiveStage("eda");
  }

  return (
    <div className="space-y-6">
      <Cell kind="explain" title="Stage 2 — Clean and prepare your data">
        <p>
          Logistic regression speaks one language: <strong>numbers on a similar scale</strong>.
          Real datasets are messy — missing cells, text categories, columns that
          range from 0–1 sitting next to columns in the thousands. Each step
          below translates raw data into something the math can chew on. Tweak
          the options and watch the "before vs after" panels react instantly.
        </p>
      </Cell>

      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Wand2 className="w-5 h-5" /> Preprocessing pipeline
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              {/* Missing values */}
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Label>Missing values</Label>
                  <Tip>
                    Real-world datasets have gaps. You either drop the row or
                    fill the gap with a sensible value (mean for numbers, mode
                    for categories).
                  </Tip>
                </div>
                <Select
                  value={preCfg.impute}
                  onValueChange={(v) =>
                    setPreCfg({ ...preCfg, impute: v as typeof preCfg.impute })
                  }
                >
                  <SelectTrigger data-testid="select-impute">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="mean">
                      Mean (numeric) / Mode (categorical)
                    </SelectItem>
                    <SelectItem value="median">
                      Median (numeric) / Mode (categorical)
                    </SelectItem>
                    <SelectItem value="mode">Mode (most frequent)</SelectItem>
                    <SelectItem value="zero">Fill with 0</SelectItem>
                    <SelectItem value="drop">
                      Drop rows with any missing feature
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Encoding */}
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Label>Categorical encoding</Label>
                  <Tip>
                    Models need numbers. <em>Label</em> assigns 0,1,2... — fine
                    for ordered categories. <em>One-hot</em> creates a separate
                    0/1 column per category — safer when there's no natural
                    order.
                  </Tip>
                </div>
                <Select
                  value={preCfg.encoding}
                  onValueChange={(v) =>
                    setPreCfg({
                      ...preCfg,
                      encoding: v as typeof preCfg.encoding,
                    })
                  }
                >
                  <SelectTrigger data-testid="select-encoding">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="onehot">One-hot encoding</SelectItem>
                    <SelectItem value="label">Label encoding</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Scaling */}
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Label>Feature scaling</Label>
                  <Tip>
                    Logistic regression is sensitive to feature magnitudes
                    because gradient descent updates all weights with the same
                    learning rate. Scaling puts every feature on the same
                    playing field.
                  </Tip>
                </div>
                <Select
                  value={preCfg.scaling}
                  onValueChange={(v) =>
                    setPreCfg({ ...preCfg, scaling: v as typeof preCfg.scaling })
                  }
                >
                  <SelectTrigger data-testid="select-scaling">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="standard">
                      Standard scaling (mean 0, std 1)
                    </SelectItem>
                    <SelectItem value="minmax">
                      Min-max scaling (range 0 to 1)
                    </SelectItem>
                    <SelectItem value="none">No scaling</SelectItem>
                  </SelectContent>
                </Select>
                {preCfg.scaling === "standard" && (
                  <div className="math-block mt-2">
                    <Tex display tex="x' = \frac{x - \mu}{\sigma}" />
                  </div>
                )}
                {preCfg.scaling === "minmax" && (
                  <div className="math-block mt-2">
                    <Tex
                      display
                      tex="x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}"
                    />
                  </div>
                )}
              </div>

              <Separator />

              {/* Feature selection */}
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <Label>Feature selection</Label>
                  <Tip>
                    Untick features you suspect are noise or leak the target.
                    Fewer, cleaner features often beat many noisy ones.
                  </Tip>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 max-h-56 overflow-y-auto pr-2">
                  {featureColumns.map((c) => {
                    const checked = preCfg.featureColumns.includes(c.name);
                    return (
                      <label
                        key={c.name}
                        className="flex items-center gap-2 rounded border p-2 cursor-pointer hover-elevate text-xs"
                      >
                        <Checkbox
                          checked={checked}
                          onCheckedChange={() => toggleFeature(c.name)}
                          data-testid={`checkbox-feature-${c.name}`}
                        />
                        <span className="font-mono truncate flex-1">
                          {c.name}
                        </span>
                        <Badge
                          variant={
                            c.type === "numeric" ? "default" : "secondary"
                          }
                          className="font-normal text-[10px] px-1"
                        >
                          {c.type === "numeric" ? "n" : "c"}
                        </Badge>
                      </label>
                    );
                  })}
                </div>
                <div className="text-xs text-muted-foreground">
                  {preCfg.featureColumns.length} of {featureColumns.length}{" "}
                  features selected
                </div>
              </div>

              <div className="flex items-center gap-3 pt-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    setPreCfg({
                      ...preCfg,
                      impute: "mean",
                      encoding: "onehot",
                      scaling: "standard",
                      featureColumns: featureColumns.map((c) => c.name),
                    })
                  }
                >
                  <RotateCcw className="w-4 h-4 mr-2" />
                  Reset to defaults
                </Button>
                <Button
                  className="ml-auto"
                  onClick={applyAndContinue}
                  disabled={!result || preCfg.featureColumns.length === 0}
                  data-testid="button-apply-preprocess"
                >
                  Apply &amp; continue
                  <ArrowRight className="w-4 h-4 ml-2" />
                </Button>
              </div>

              {error && (
                <div className="text-sm text-destructive">{error}</div>
              )}
            </CardContent>
          </Card>

          {result && (
            <Card>
              <CardHeader>
                <CardTitle>Before vs After</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-3 gap-3">
                  <Stat
                    label="Rows"
                    before={result.beforeAfter.rowsBefore}
                    after={result.beforeAfter.rowsAfter}
                  />
                  <Stat
                    label="Missing cells"
                    before={result.beforeAfter.missingBefore}
                    after={result.beforeAfter.missingAfter}
                  />
                  <Stat
                    label="Feature columns"
                    before={result.beforeAfter.featureCountBefore}
                    after={result.beforeAfter.featureCountAfter}
                  />
                </div>

                <div className="grid lg:grid-cols-2 gap-4">
                  <div>
                    <h5 className="text-xs font-medium text-muted-foreground mb-1">
                      Before (raw)
                    </h5>
                    <div className="overflow-x-auto rounded border max-h-64">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            {dataset.columns.slice(0, 5).map((c) => (
                              <TableHead
                                key={c.name}
                                className="font-mono text-[10px]"
                              >
                                {c.name}
                              </TableHead>
                            ))}
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {result.beforeAfter.sampleBefore.map((r, i) => (
                            <TableRow key={i}>
                              {dataset.columns.slice(0, 5).map((c) => (
                                <TableCell
                                  key={c.name}
                                  className="font-mono text-[10px] tabular-nums"
                                >
                                  {r[c.name] === null
                                    ? "—"
                                    : String(r[c.name])}
                                </TableCell>
                              ))}
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </div>
                  <div>
                    <h5 className="text-xs font-medium text-muted-foreground mb-1">
                      After (model-ready)
                    </h5>
                    <div className="overflow-x-auto rounded border max-h-64">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            {result.processed.featureNames
                              .slice(0, 5)
                              .map((n) => (
                                <TableHead
                                  key={n}
                                  className="font-mono text-[10px]"
                                >
                                  {n}
                                </TableHead>
                              ))}
                            <TableHead className="font-mono text-[10px]">
                              y
                            </TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {result.beforeAfter.sampleAfter.map((r, i) => (
                            <TableRow key={i}>
                              {result.processed.featureNames
                                .slice(0, 5)
                                .map((n) => (
                                  <TableCell
                                    key={n}
                                    className="font-mono text-[10px] tabular-nums"
                                  >
                                    {fmt(r[n] ?? 0, 3)}
                                  </TableCell>
                                ))}
                              <TableCell className="font-mono text-[10px] tabular-nums font-bold">
                                {r["__y__"]}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </div>
                </div>

                <ul className="text-xs text-muted-foreground space-y-1 list-disc list-inside">
                  {result.processed.notes.map((n, i) => (
                    <li key={i}>{n}</li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}
        </div>

        <div className="space-y-6">
          <Cell kind="code" title="Equivalent Python (sklearn)">
            <CodeBlock>
              {`from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

pre = ColumnTransformer([
  ("num", make_pipeline(
      SimpleImputer(strategy="${preCfg.impute === "drop" ? "mean" : preCfg.impute}"),
      ${preCfg.scaling === "standard" ? "StandardScaler()" : preCfg.scaling === "minmax" ? "MinMaxScaler()" : '"passthrough"'}
  ), num_cols),
  ("cat", ${preCfg.encoding === "onehot" ? "OneHotEncoder(handle_unknown='ignore')" : "OrdinalEncoder()"}, cat_cols)
])
X = pre.fit_transform(df[features])`}
            </CodeBlock>
          </Cell>

          <Cell kind="explain" title="Why scaling matters">
            <p>
              Imagine one feature is age (0–100) and another is income
              (0–200,000). The income feature would dominate the gradient and
              the model would basically ignore age. After standard scaling both
              features have <Tex tex="\mu=0" /> and{" "}
              <Tex tex="\sigma=1" /> — they get equal say.
            </p>
          </Cell>

          {processed && processed === result?.processed && (
            <Cell kind="viz" title="Ready">
              <p className="text-emerald-700 dark:text-emerald-400">
                Pipeline applied. The next step is to look around the data
                before training.
              </p>
            </Cell>
          )}
        </div>
      </div>
    </div>
  );
}

function Stat({
  label,
  before,
  after,
}: {
  label: string;
  before: number;
  after: number;
}) {
  const changed = before !== after;
  return (
    <div className="rounded border p-3 bg-muted/30">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div className="flex items-baseline gap-2 mt-1">
        <span className="text-sm text-muted-foreground line-through">
          {before}
        </span>
        <ArrowRight className="w-3 h-3 text-muted-foreground" />
        <span
          className={`text-lg font-semibold tabular-nums ${changed ? "text-primary" : ""}`}
        >
          {after}
        </span>
      </div>
    </div>
  );
}
