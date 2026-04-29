import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RTooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceDot,
  Area,
  AreaChart,
} from "recharts";
import { Cell, CodeBlock } from "@/components/Notebook";
import { Math as Tex } from "@/components/Math";
import { sigmoid, fmt, dot } from "@/lib/math-utils";
import { useStore } from "@/lib/store";
import { ArrowRight } from "lucide-react";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export function StageTheory() {
  const { setActiveStage, processed } = useStore();
  const [zPicker, setZPicker] = useState(0);

  // Sigmoid curve
  const sigmoidData = useMemo(() => {
    const out: { z: number; sigma: number }[] = [];
    for (let z = -8; z <= 8; z += 0.1) out.push({ z, sigma: sigmoid(z) });
    return out;
  }, []);

  // Pick first sample for walkthrough (if processed exists)
  const walkthrough = useMemo(() => {
    if (!processed || processed.featureMatrix.length === 0) return null;
    const x = processed.featureMatrix[0];
    const fakeWeights = processed.featureNames.map(
      (_, i) => 0.5 - (i % 3) * 0.3,
    );
    const bias = -0.5;
    const z = dot(x, fakeWeights) + bias;
    const p = sigmoid(z);
    return { x, weights: fakeWeights, bias, z, p };
  }, [processed]);

  return (
    <div className="space-y-6">
      <Cell kind="explain" title="Stage 4 — How logistic regression actually works">
        <p>
          Logistic regression is misnamed — it's a <strong>classifier</strong>,
          not a regressor. The trick is two-step: build a linear score, then
          squash it into a probability between 0 and 1. Pick a threshold and
          you get a class label.
        </p>
      </Cell>

      <Tabs defaultValue="binary">
        <TabsList>
          <TabsTrigger value="binary">Binary</TabsTrigger>
          <TabsTrigger value="multi">Multi-class (One-vs-Rest)</TabsTrigger>
          <TabsTrigger value="vs">Linear vs Logistic</TabsTrigger>
        </TabsList>

        <TabsContent value="binary" className="space-y-6 pt-4">
          <div className="grid lg:grid-cols-2 gap-6">
            <Cell kind="math" title="Step 1 — The linear score">
              <p>
                Each feature gets a weight. Multiply, add them up, add a bias.
                Just like a linear equation:
              </p>
              <div className="math-block mt-3">
                <Tex
                  display
                  tex="z = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b"
                />
              </div>
              <p className="mt-3 text-muted-foreground text-xs">
                Big positive <Tex tex="z" />: model leans toward class 1. Big
                negative: leans toward class 0. Zero: it's a coin-flip.
              </p>
            </Cell>

            <Cell kind="math" title="Step 2 — Squash into a probability">
              <p>
                The problem with <Tex tex="z" />: it can be anything from{" "}
                <Tex tex="-\infty" /> to <Tex tex="+\infty" />. Probabilities
                have to live between 0 and 1. The <strong>sigmoid</strong>{" "}
                function does the squashing:
              </p>
              <div className="math-block mt-3">
                <Tex display tex="\sigma(z) = \frac{1}{1 + e^{-z}}" />
              </div>
              <p className="mt-3 text-muted-foreground text-xs">
                <Tex tex="\sigma(0) = 0.5" />, <Tex tex="\sigma(\infty)=1" />,{" "}
                <Tex tex="\sigma(-\infty)=0" />. Smooth, monotonic,
                differentiable — perfect for gradient descent.
              </p>
            </Cell>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                Play with the sigmoid
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid lg:grid-cols-3 gap-6 items-start">
                <div className="lg:col-span-2">
                  <ResponsiveContainer width="100%" height={280}>
                    <AreaChart data={sigmoidData}>
                      <defs>
                        <linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">
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
                        dataKey="z"
                        type="number"
                        domain={[-8, 8]}
                        tick={{ fontSize: 11 }}
                        label={{
                          value: "z (linear score)",
                          position: "insideBottom",
                          offset: -5,
                          fontSize: 11,
                        }}
                      />
                      <YAxis
                        domain={[0, 1]}
                        tick={{ fontSize: 11 }}
                        label={{
                          value: "σ(z)",
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
                      <Area
                        type="monotone"
                        dataKey="sigma"
                        stroke="hsl(var(--chart-1))"
                        strokeWidth={2}
                        fill="url(#sg)"
                        isAnimationActive={false}
                      />
                      <ReferenceLine
                        y={0.5}
                        stroke="hsl(var(--muted-foreground))"
                        strokeDasharray="4 4"
                        label={{
                          value: "decision threshold = 0.5",
                          fontSize: 10,
                          fill: "hsl(var(--muted-foreground))",
                          position: "insideTopRight",
                        }}
                      />
                      <ReferenceLine
                        x={0}
                        stroke="hsl(var(--muted-foreground))"
                        strokeDasharray="2 2"
                      />
                      <ReferenceDot
                        x={zPicker}
                        y={sigmoid(zPicker)}
                        r={6}
                        fill="hsl(var(--accent))"
                        stroke="white"
                        strokeWidth={2}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <div className="space-y-4">
                  <div>
                    <label className="text-xs text-muted-foreground">
                      Drag z
                    </label>
                    <Slider
                      value={[zPicker]}
                      onValueChange={(v) => setZPicker(v[0])}
                      min={-8}
                      max={8}
                      step={0.1}
                      data-testid="slider-z"
                    />
                  </div>
                  <div className="rounded-md border bg-muted/30 p-3 font-mono text-sm space-y-1">
                    <div>
                      z = <span className="text-primary">{fmt(zPicker, 2)}</span>
                    </div>
                    <div>
                      σ(z) ={" "}
                      <span className="text-accent font-bold">
                        {fmt(sigmoid(zPicker), 4)}
                      </span>
                    </div>
                    <div className="text-xs text-muted-foreground pt-1">
                      Predicted class:{" "}
                      <span className="text-foreground font-bold">
                        {sigmoid(zPicker) >= 0.5 ? "1" : "0"}
                      </span>
                    </div>
                  </div>
                  <div className="rounded-md border-l-2 border-amber-500 bg-amber-50 dark:bg-amber-950/30 p-3 text-xs">
                    <p className="font-medium mb-1">Why a threshold of 0.5?</p>
                    <p>
                      0.5 is the natural midpoint of the sigmoid — at z = 0 the
                      model has no preference. You can move the threshold to
                      trade off precision vs recall (more on that during
                      evaluation).
                    </p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Cell kind="math" title="Step 3 — How parameters get learned">
            <p>
              Define a loss that punishes wrong-confidence:{" "}
              <strong>binary cross-entropy</strong>:
            </p>
            <div className="math-block mt-3">
              <Tex
                display
                tex="\mathcal{L} = -\frac{1}{n}\sum_i \big[y_i\log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\big]"
              />
            </div>
            <p className="mt-3">
              Take the gradient of the loss with respect to each weight, nudge
              the weights opposite the gradient, repeat. That's gradient
              descent. The gradient is conveniently clean:
            </p>
            <div className="math-block mt-3">
              <Tex
                display
                tex="\frac{\partial \mathcal{L}}{\partial w_j} = \frac{1}{n}\sum_i (\hat{p}_i - y_i)\, x_{ij}"
              />
            </div>
          </Cell>

          {walkthrough && processed && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">
                  Walkthrough: predict the first sample
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3 text-sm">
                <p className="text-muted-foreground">
                  Using <em>illustrative</em> weights (we haven't trained yet),
                  here's exactly what happens for sample #0 of your processed
                  data:
                </p>
                <div className="overflow-x-auto rounded border max-h-56">
                  <table className="w-full text-xs font-mono">
                    <thead className="bg-muted">
                      <tr>
                        <th className="text-left p-2">feature</th>
                        <th className="text-right p-2">x</th>
                        <th className="text-right p-2">w</th>
                        <th className="text-right p-2">w · x</th>
                      </tr>
                    </thead>
                    <tbody>
                      {processed.featureNames.slice(0, 8).map((n, i) => (
                        <tr key={n} className="border-t">
                          <td className="p-2">{n}</td>
                          <td className="p-2 text-right tabular-nums">
                            {fmt(walkthrough.x[i], 3)}
                          </td>
                          <td className="p-2 text-right tabular-nums">
                            {fmt(walkthrough.weights[i], 3)}
                          </td>
                          <td className="p-2 text-right tabular-nums">
                            {fmt(
                              walkthrough.x[i] * walkthrough.weights[i],
                              3,
                            )}
                          </td>
                        </tr>
                      ))}
                      {processed.featureNames.length > 8 && (
                        <tr>
                          <td
                            colSpan={4}
                            className="text-center text-muted-foreground p-2"
                          >
                            …{processed.featureNames.length - 8} more features
                          </td>
                        </tr>
                      )}
                      <tr className="border-t bg-muted/30">
                        <td className="p-2 font-bold">+ bias</td>
                        <td colSpan={2}></td>
                        <td className="p-2 text-right tabular-nums font-bold">
                          {fmt(walkthrough.bias, 3)}
                        </td>
                      </tr>
                      <tr className="border-t bg-muted/50">
                        <td className="p-2 font-bold">= z</td>
                        <td colSpan={2}></td>
                        <td className="p-2 text-right tabular-nums font-bold text-primary">
                          {fmt(walkthrough.z, 3)}
                        </td>
                      </tr>
                    </tbody>
                  </table>
                </div>
                <div className="rounded-md bg-muted/30 border p-3 font-mono text-sm space-y-1">
                  <div>
                    σ(z) = σ({fmt(walkthrough.z, 3)}) ={" "}
                    <span className="text-accent font-bold">
                      {fmt(walkthrough.p, 4)}
                    </span>
                  </div>
                  <div className="text-xs text-muted-foreground pt-1">
                    Threshold 0.5 → predicted class ={" "}
                    <span className="text-foreground font-bold">
                      {walkthrough.p >= 0.5 ? "1" : "0"}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="multi" className="space-y-4 pt-4">
          <Cell kind="explain" title="Multi-class via One-vs-Rest">
            <p>
              Sigmoid only gives you one probability — class 1 or "not class 1".
              For 3+ classes, the simplest extension is <strong>One-vs-Rest</strong>{" "}
              (OvR):
            </p>
            <ol className="list-decimal ml-6 mt-2 space-y-1 text-sm">
              <li>
                Train one binary classifier per class: "is this class A or
                anything else?"
              </li>
              <li>
                At prediction time, run all <Tex tex="K" /> classifiers, get{" "}
                <Tex tex="K" /> raw scores <Tex tex="z_k" />.
              </li>
              <li>
                Normalize them with <strong>softmax</strong> so they sum to 1:
              </li>
            </ol>
            <div className="math-block mt-3">
              <Tex
                display
                tex="\hat{p}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}"
              />
            </div>
            <p className="mt-2 text-muted-foreground text-xs">
              This lab automatically uses OvR + softmax when your target has more
              than two classes.
            </p>
          </Cell>
        </TabsContent>

        <TabsContent value="vs" className="space-y-4 pt-4">
          <div className="grid md:grid-cols-2 gap-4">
            <Cell kind="explain" title="Linear regression">
              <ul className="space-y-2 text-sm">
                <li>
                  <strong>Output:</strong> a real number (price, height,
                  temperature).
                </li>
                <li>
                  <strong>Loss:</strong> mean squared error.
                </li>
                <li>
                  <strong>What it predicts:</strong> the value itself.
                </li>
                <li>
                  <strong>Threshold:</strong> none — there's nothing to
                  classify.
                </li>
              </ul>
            </Cell>
            <Cell kind="explain" title="Logistic regression">
              <ul className="space-y-2 text-sm">
                <li>
                  <strong>Output:</strong> a probability between 0 and 1.
                </li>
                <li>
                  <strong>Loss:</strong> binary cross-entropy.
                </li>
                <li>
                  <strong>What it predicts:</strong> the chance that the sample
                  belongs to the positive class.
                </li>
                <li>
                  <strong>Threshold:</strong> apply one (typically 0.5) to get a
                  class label.
                </li>
              </ul>
            </Cell>
          </div>
          <Cell kind="explain" title="Probability vs class prediction">
            <p>
              The model's <em>raw</em> output is a probability — that's the
              richer information. The <em>class</em> is just a categorical
              decision you make on top: "if probability ≥ threshold, call it
              class 1." Two predictions of 0.51 and 0.99 both get classified as
              "1" but they're not equally confident — keep the probabilities
              when you can.
            </p>
          </Cell>
        </TabsContent>
      </Tabs>

      <Cell kind="code" title="The whole thing in code">
        <CodeBlock>
          {`import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def fit(X, y, lr=0.1, epochs=200, l2=0.01):
    n, d = X.shape
    w, b = np.zeros(d), 0.0
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        grad_w = (X.T @ (p - y)) / n + l2 * w
        grad_b = (p - y).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

def predict_proba(X, w, b):
    return sigmoid(X @ w + b)

def predict(X, w, b, threshold=0.5):
    return (predict_proba(X, w, b) >= threshold).astype(int)`}
        </CodeBlock>
      </Cell>

      <div className="flex justify-end">
        <Button
          onClick={() => setActiveStage("train")}
          data-testid="button-continue-theory"
        >
          Train the model
          <ArrowRight className="w-4 h-4 ml-2" />
        </Button>
      </div>
    </div>
  );
}
