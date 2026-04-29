# Workspace

## Overview

pnpm workspace monorepo using TypeScript. Each package manages its own dependencies.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.

## Artifacts

### `artifacts/logreg-lab` — Logistic Regression Learning Lab

Interactive, notebook-style web app that teaches logistic regression end-to-end. Runs entirely client-side (no backend, no API). 7-stage pipeline navigated via a left sidebar with progressive unlock based on prior-stage completion.

Stages:
1. **Data** — CSV upload (drag/drop) or built-in Pima Indians Diabetes sample. Schema preview, target column picker, target distribution.
2. **Preprocess** — missing-value imputation, categorical encoding (label / one-hot), scaling (standard / min-max / none), feature selection. Live before/after panels and equivalent sklearn snippet.
3. **EDA** — class distribution bar, correlation-with-target ranking, per-feature histogram split by class, feature-vs-target jittered scatter.
4. **Theory** — sigmoid playground with draggable z, binary cross-entropy + gradient derivation, walkthrough of one real sample (z = w·x + b → σ(z) → class), tabs for Binary / OvR multi-class / Linear-vs-Logistic.
5. **Train** — sliders for learning rate, epochs, L2, decision threshold, train/test split, optional k-fold CV. Loss curve, learned weights bars, 2-feature decision boundary scatter, probability curve along a chosen feature.
6. **Evaluate** — accuracy / macro-F1 / AUC / overfit warning KPIs, color-graded confusion matrix, per-class precision/recall/F1 table, ROC curve with AUC fill, cross-validation fold scores.
7. **Predict** — input form prefilled from a dataset row (or random/select), step-by-step contributions table, z → σ(z) → class with rendered KaTeX equations, probability gauge with threshold marker. OvR multi-class shows per-class softmax bars.

Stack: React 19 + Vite, Recharts (charts), Papaparse (CSV), KaTeX (math typesetting), shadcn/ui primitives, Tailwind. All ML (training, prediction, metrics) implemented from scratch in TypeScript under `src/lib/` (`logreg.ts`, `metrics.ts`, `preprocess.ts`, `math-utils.ts`). State managed by a single `StoreProvider` context (`src/lib/store.tsx`).

Note: the imported `Math` KaTeX component is aliased as `Tex` in stage files to avoid shadowing the global `Math` object.
