import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { StoreProvider, useStore, type StageId } from "@/lib/store";
import { StageData } from "@/components/stages/StageData";
import { StagePreprocess } from "@/components/stages/StagePreprocess";
import { StageEDA } from "@/components/stages/StageEDA";
import { StageTheory } from "@/components/stages/StageTheory";
import { StageTrain } from "@/components/stages/StageTrain";
import { StageEvaluate } from "@/components/stages/StageEvaluate";
import { StagePredict } from "@/components/stages/StagePredict";
import {
  Database,
  Wand2,
  BarChart3,
  BookOpen,
  Sparkles,
  Activity,
  Target,
  CircleDot,
  CheckCircle2,
  Lock,
} from "lucide-react";
import type { ComponentType } from "react";

const queryClient = new QueryClient();

interface StageSpec {
  id: StageId;
  num: number;
  title: string;
  subtitle: string;
  icon: ComponentType<{ className?: string }>;
}

const STAGES: StageSpec[] = [
  {
    id: "data",
    num: 1,
    title: "Data",
    subtitle: "Load CSV, pick target",
    icon: Database,
  },
  {
    id: "preprocess",
    num: 2,
    title: "Preprocess",
    subtitle: "Clean, encode, scale",
    icon: Wand2,
  },
  {
    id: "eda",
    num: 3,
    title: "Explore",
    subtitle: "Distributions & correlations",
    icon: BarChart3,
  },
  {
    id: "theory",
    num: 4,
    title: "Theory",
    subtitle: "Sigmoid, threshold, loss",
    icon: BookOpen,
  },
  {
    id: "train",
    num: 5,
    title: "Train",
    subtitle: "Fit + visualise",
    icon: Activity,
  },
  {
    id: "evaluate",
    num: 6,
    title: "Evaluate",
    subtitle: "Confusion, ROC, CV",
    icon: Target,
  },
  {
    id: "predict",
    num: 7,
    title: "Predict",
    subtitle: "Step-by-step inference",
    icon: Sparkles,
  },
];

function Sidebar() {
  const { activeStage, setActiveStage, dataset, processed, training } =
    useStore();

  function isUnlocked(id: StageId): boolean {
    if (id === "data") return true;
    if (id === "preprocess") return !!dataset;
    if (id === "eda" || id === "theory") return !!processed;
    if (id === "train") return !!processed;
    if (id === "evaluate" || id === "predict") return !!training;
    return false;
  }

  function isComplete(id: StageId): boolean {
    if (id === "data") return !!dataset;
    if (id === "preprocess") return !!processed;
    if (id === "eda") return !!processed; // visiting it is enough
    if (id === "theory") return false;
    if (id === "train") return !!training;
    return false;
  }

  return (
    <aside className="w-72 border-r bg-card/40 backdrop-blur-sm flex flex-col h-screen sticky top-0">
      <div className="p-5 border-b">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-primary to-accent flex items-center justify-center text-white font-bold">
            σ
          </div>
          <div>
            <h1 className="font-semibold text-sm leading-tight">
              Logistic Regression
            </h1>
            <p className="text-[11px] text-muted-foreground leading-tight">
              Interactive Learning Lab
            </p>
          </div>
        </div>
      </div>

      <nav className="flex-1 overflow-y-auto p-3 space-y-1">
        {STAGES.map((s) => {
          const unlocked = isUnlocked(s.id);
          const active = activeStage === s.id;
          const done = isComplete(s.id);
          const Icon = s.icon;
          return (
            <button
              key={s.id}
              onClick={() => unlocked && setActiveStage(s.id)}
              disabled={!unlocked}
              data-testid={`nav-${s.id}`}
              className={`w-full text-left px-3 py-2.5 rounded-md flex items-start gap-3 transition-colors ${
                active
                  ? "bg-primary/10 text-foreground"
                  : unlocked
                    ? "hover-elevate text-foreground"
                    : "text-muted-foreground/50 cursor-not-allowed"
              }`}
            >
              <div
                className={`mt-0.5 w-7 h-7 rounded-md flex items-center justify-center text-xs font-semibold flex-shrink-0 ${
                  active
                    ? "bg-primary text-primary-foreground"
                    : done
                      ? "bg-emerald-500/20 text-emerald-700 dark:text-emerald-400"
                      : "bg-muted text-muted-foreground"
                }`}
              >
                {!unlocked ? (
                  <Lock className="w-3 h-3" />
                ) : done && !active ? (
                  <CheckCircle2 className="w-3.5 h-3.5" />
                ) : (
                  <Icon className="w-3.5 h-3.5" />
                )}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className="text-[10px] font-mono text-muted-foreground">
                    0{s.num}
                  </span>
                  <span className="text-sm font-medium truncate">
                    {s.title}
                  </span>
                  {active && (
                    <CircleDot className="w-3 h-3 text-primary ml-auto flex-shrink-0" />
                  )}
                </div>
                <p className="text-[11px] text-muted-foreground leading-tight mt-0.5">
                  {s.subtitle}
                </p>
              </div>
            </button>
          );
        })}
      </nav>

      <div className="p-3 border-t text-[11px] text-muted-foreground space-y-1">
        <p>
          A notebook-style sandbox for learning logistic regression by doing.
        </p>
        <p className="text-[10px]">
          Build with React • runs entirely in your browser.
        </p>
      </div>
    </aside>
  );
}

function ActiveStage() {
  const { activeStage } = useStore();
  switch (activeStage) {
    case "data":
      return <StageData />;
    case "preprocess":
      return <StagePreprocess />;
    case "eda":
      return <StageEDA />;
    case "theory":
      return <StageTheory />;
    case "train":
      return <StageTrain />;
    case "evaluate":
      return <StageEvaluate />;
    case "predict":
      return <StagePredict />;
    default:
      return null;
  }
}

function StageHeader() {
  const { activeStage, dataset } = useStore();
  const stage = STAGES.find((s) => s.id === activeStage);
  if (!stage) return null;
  return (
    <div className="border-b bg-background/80 backdrop-blur-sm sticky top-0 z-10">
      <div className="px-8 py-4 flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">
            <span className="font-mono">Stage {stage.num} / 7</span>
            <span>·</span>
            <span>{stage.subtitle}</span>
          </div>
          <h2 className="text-2xl font-semibold mt-0.5">{stage.title}</h2>
        </div>
        {dataset && (
          <div className="hidden md:flex items-center gap-2 text-xs text-muted-foreground">
            <Database className="w-3.5 h-3.5" />
            <span className="font-mono">{dataset.fileName}</span>
            <span>·</span>
            <span>
              {dataset.rowCount} rows × {dataset.columns.length} cols
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function Shell() {
  return (
    <div className="flex bg-background min-h-screen">
      <Sidebar />
      <main className="flex-1 min-w-0">
        <StageHeader />
        <div className="p-6 lg:p-8 max-w-6xl mx-auto">
          <ActiveStage />
        </div>
      </main>
    </div>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider delayDuration={150}>
        <StoreProvider>
          <Shell />
        </StoreProvider>
        <Toaster />
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
