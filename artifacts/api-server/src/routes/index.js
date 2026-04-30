import { Router } from "express";
import healthRouter from "./health.js";

const router = Router();

// Mount health routes
router.use("/health", healthRouter);

// Add other route groups here as needed

export default router;
