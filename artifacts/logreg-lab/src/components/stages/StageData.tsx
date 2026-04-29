import { useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Upload, FileText, ArrowRight, Database } from "lucide-react";
import { parseCsv, SAMPLE_DATASET_CSV } from "@/lib/dataset";
import type { Dataset } from "@/lib/dataset";
import { useStore } from "@/lib/store";
import { Cell } from "@/components/Notebook";
import { useToast } from "@/hooks/use-toast";

export function StageData() {
  const {
    dataset,
    setDataset,
    setPreCfg,
    setProcessed,
    setTraining,
    setActiveStage,
  } = useStore();
  const inputRef = useRef<HTMLInputElement>(null);
  const [target, setTarget] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const { toast } = useToast();

  async function handleFile(file: File) {
    setBusy(true);
    try {
      const ds = await parseCsv(file);
      finishLoad(ds);
    } catch (e) {
      toast({
        title: "Failed to read CSV",
        description: e instanceof Error ? e.message : String(e),
        variant: "destructive",
      });
    } finally {
      setBusy(false);
    }
  }

  async function loadSample() {
    setBusy(true);
    try {
      const file = new File([SAMPLE_DATASET_CSV], "diabetes.csv", {
        type: "text/csv",
      });
      const ds = await parseCsv(file);
      finishLoad(ds);
    } finally {
      setBusy(false);
    }
  }

  function finishLoad(ds: Dataset) {
    setDataset(ds);
    setProcessed(null);
    setTraining(null);
    setPreCfg(null);
    // Auto-pick a likely binary target
    const candidate =
      ds.columns.find(
        (c) =>
          c.uniqueCount === 2 &&
          /target|outcome|label|class|diagnosis|y/i.test(c.name),
      ) ??
      ds.columns.find((c) => c.uniqueCount === 2) ??
      ds.columns[ds.columns.length - 1];
    setTarget(candidate?.name ?? null);
  }

  function confirmAndContinue() {
    if (!dataset || !target) return;
    setPreCfg({
      targetColumn: target,
      featureColumns: dataset.columns
        .filter((c) => c.name !== target)
        .map((c) => c.name),
      impute: "mean",
      encoding: "onehot",
      scaling: "standard",
    });
    setActiveStage("preprocess");
  }

  return (
    <div className="space-y-6">
      <Cell kind="explain" title="Stage 1 — Load your dataset">
        <p>
          A model only knows what you show it. Start by giving it some
          examples — a CSV file where each row is one observation and each column
          is a feature. One column must be the <strong>target</strong>: the thing
          you want the model to predict (e.g. "diabetic" yes/no).
        </p>
        <p className="mt-2 text-muted-foreground">
          Logistic regression is a <em>classification</em> algorithm — the target
          should be categorical (binary or a small number of classes), not a
          continuous number.
        </p>
      </Cell>

      {!dataset && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="w-5 h-5" /> Upload a CSV
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div
              className="border-2 border-dashed rounded-lg p-10 text-center hover-elevate cursor-pointer"
              onClick={() => inputRef.current?.click()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                const f = e.dataTransfer.files?.[0];
                if (f) void handleFile(f);
              }}
            >
              <Upload className="w-8 h-8 mx-auto text-muted-foreground" />
              <p className="mt-3 font-medium">
                Drop a CSV file here or click to browse
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                Header row required • numeric or categorical columns • binary or
                multi-class target
              </p>
              <input
                ref={inputRef}
                type="file"
                accept=".csv,text/csv"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) void handleFile(f);
                }}
              />
            </div>
            <div className="flex items-center gap-3">
              <div className="flex-1 h-px bg-border" />
              <span className="text-xs text-muted-foreground">or</span>
              <div className="flex-1 h-px bg-border" />
            </div>
            <Button
              variant="outline"
              onClick={() => void loadSample()}
              disabled={busy}
              className="w-full"
              data-testid="button-load-sample"
            >
              <Database className="w-4 h-4 mr-2" />
              Load sample dataset (Pima Indians Diabetes — 290 rows)
            </Button>
          </CardContent>
        </Card>
      )}

      {dataset && (
        <>
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="w-5 h-5" />
                {dataset.fileName}
                <Badge variant="secondary" className="ml-auto">
                  {dataset.rowCount} rows × {dataset.columns.length} cols
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <h4 className="text-sm font-medium mb-2">Schema</h4>
                <div className="overflow-x-auto rounded border">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Column</TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead className="text-right">Unique</TableHead>
                        <TableHead className="text-right">Missing</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {dataset.columns.map((c) => (
                        <TableRow key={c.name}>
                          <TableCell className="font-mono text-xs">
                            {c.name}
                          </TableCell>
                          <TableCell>
                            <Badge
                              variant={c.type === "numeric" ? "default" : "secondary"}
                              className="font-normal"
                            >
                              {c.type}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-right text-xs tabular-nums">
                            {c.uniqueCount}
                          </TableCell>
                          <TableCell className="text-right text-xs tabular-nums">
                            {c.missing > 0 ? (
                              <span className="text-destructive">
                                {c.missing}
                              </span>
                            ) : (
                              "0"
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>

              <div>
                <h4 className="text-sm font-medium mb-2">
                  Preview (first 5 rows)
                </h4>
                <div className="overflow-x-auto rounded border">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        {dataset.columns.map((c) => (
                          <TableHead key={c.name} className="font-mono text-xs">
                            {c.name}
                          </TableHead>
                        ))}
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {dataset.rows.slice(0, 5).map((r, i) => (
                        <TableRow key={i}>
                          {dataset.columns.map((c) => (
                            <TableCell
                              key={c.name}
                              className="font-mono text-xs tabular-nums"
                            >
                              {r[c.name] === null ? (
                                <span className="text-muted-foreground italic">
                                  null
                                </span>
                              ) : (
                                String(r[c.name])
                              )}
                            </TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Pick the target column</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-sm text-muted-foreground">
                The target is what the model will learn to predict. Choose a
                column with 2 (binary) or a few discrete values (multi-class).
              </p>
              <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-end">
                <div className="flex-1 w-full">
                  <Select
                    value={target ?? ""}
                    onValueChange={(v) => setTarget(v)}
                  >
                    <SelectTrigger data-testid="select-target">
                      <SelectValue placeholder="Select target column" />
                    </SelectTrigger>
                    <SelectContent>
                      {dataset.columns
                        .filter((c) => c.name !== "")
                        .map((c) => (
                          <SelectItem key={c.name} value={c.name}>
                            {c.name} — {c.uniqueCount} unique
                          </SelectItem>
                        ))}
                    </SelectContent>
                  </Select>
                </div>
                <Button
                  onClick={confirmAndContinue}
                  disabled={!target}
                  data-testid="button-continue-data"
                >
                  Continue to preprocessing
                  <ArrowRight className="w-4 h-4 ml-2" />
                </Button>
              </div>
              {target && (
                <TargetSummary dataset={dataset} target={target} />
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setDataset(null);
                  setTarget(null);
                }}
              >
                Use a different file
              </Button>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

function TargetSummary({
  dataset,
  target,
}: {
  dataset: Dataset;
  target: string;
}) {
  const counts = new Map<string, number>();
  for (const r of dataset.rows) {
    const v = r[target];
    if (v === null || v === undefined) continue;
    const k = String(v);
    counts.set(k, (counts.get(k) ?? 0) + 1);
  }
  const total = Array.from(counts.values()).reduce((s, v) => s + v, 0);
  const entries = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
  const isBinary = entries.length === 2;
  return (
    <div className="rounded-md border bg-muted/30 p-3">
      <div className="flex items-center gap-2 text-sm">
        <Badge variant={isBinary ? "default" : "secondary"}>
          {isBinary ? "Binary" : `${entries.length}-class`}
        </Badge>
        <span className="text-muted-foreground">
          target distribution:
        </span>
      </div>
      <div className="mt-2 space-y-1">
        {entries.map(([k, c]) => (
          <div key={k} className="flex items-center gap-2 text-xs">
            <span className="font-mono w-24 truncate">{k}</span>
            <div className="flex-1 h-2 bg-secondary rounded overflow-hidden">
              <div
                className="h-full bg-primary"
                style={{ width: `${(c / total) * 100}%` }}
              />
            </div>
            <span className="tabular-nums w-20 text-right">
              {c} ({((c / total) * 100).toFixed(1)}%)
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
