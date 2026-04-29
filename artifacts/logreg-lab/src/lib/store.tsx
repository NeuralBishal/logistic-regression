import {
  createContext,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { Dataset } from "./dataset";
import type {
  PreprocessConfig,
  ProcessedDataset,
} from "./preprocess";
import type { LogRegModel, TrainConfig } from "./logreg";
import { DEFAULT_TRAIN_CONFIG } from "./logreg";
import type { EvaluationResult } from "./metrics";

export type StageId =
  | "data"
  | "preprocess"
  | "eda"
  | "theory"
  | "train"
  | "evaluate"
  | "predict"
  | "compare";

export interface TrainingArtifacts {
  model: LogRegModel;
  Xtrain: number[][];
  ytrain: number[];
  Xtest: number[][];
  ytest: number[];
  testEval: EvaluationResult;
  trainEval: EvaluationResult;
  cv?: { foldScores: number[]; mean: number; std: number; k: number };
}

interface StoreState {
  dataset: Dataset | null;
  setDataset: (d: Dataset | null) => void;
  preCfg: PreprocessConfig | null;
  setPreCfg: (c: PreprocessConfig | null) => void;
  processed: ProcessedDataset | null;
  setProcessed: (p: ProcessedDataset | null) => void;
  trainCfg: TrainConfig;
  setTrainCfg: (c: TrainConfig) => void;
  testSize: number;
  setTestSize: (n: number) => void;
  useCV: boolean;
  setUseCV: (b: boolean) => void;
  cvK: number;
  setCvK: (n: number) => void;
  training: TrainingArtifacts | null;
  setTraining: (t: TrainingArtifacts | null) => void;
  activeStage: StageId;
  setActiveStage: (s: StageId) => void;
}

const StoreCtx = createContext<StoreState | null>(null);

export function StoreProvider({ children }: { children: ReactNode }) {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [preCfg, setPreCfg] = useState<PreprocessConfig | null>(null);
  const [processed, setProcessed] = useState<ProcessedDataset | null>(null);
  const [trainCfg, setTrainCfg] = useState<TrainConfig>(DEFAULT_TRAIN_CONFIG);
  const [testSize, setTestSize] = useState(0.2);
  const [useCV, setUseCV] = useState(false);
  const [cvK, setCvK] = useState(5);
  const [training, setTraining] = useState<TrainingArtifacts | null>(null);
  const [activeStage, setActiveStage] = useState<StageId>("data");

  const value = useMemo(
    () => ({
      dataset,
      setDataset,
      preCfg,
      setPreCfg,
      processed,
      setProcessed,
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
      activeStage,
      setActiveStage,
    }),
    [
      dataset,
      preCfg,
      processed,
      trainCfg,
      testSize,
      useCV,
      cvK,
      training,
      activeStage,
    ],
  );

  return <StoreCtx.Provider value={value}>{children}</StoreCtx.Provider>;
}

export function useStore() {
  const v = useContext(StoreCtx);
  if (!v) throw new Error("useStore must be used inside StoreProvider");
  return v;
}
