import crypto from "crypto";
import fs from "fs";
import path from "path";

import express from "express";
import multer from "multer";

import { getQueueSnapshot, enqueueJob, getJob } from "../services/jobQueue.js";

const uploadsDir = path.resolve("uploads");
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

const upload = multer({
  dest: uploadsDir,
  limits: {
    fileSize: 5 * 1024 * 1024,
  },
  fileFilter: (_req, file, cb) => {
    if (file.mimetype === "text/csv" || file.originalname.toLowerCase().endsWith(".csv")) {
      cb(null, true);
      return;
    }
    cb(new Error("Only CSV files are supported"));
  },
});

const jobsRouter = express.Router();

jobsRouter.post("/upload", upload.single("file"), (req, res) => {
  if (!req.file) {
    res.status(400).json({ error: "CSV file is required" });
    return;
  }

  const jobId = crypto.randomUUID();

  enqueueJob({
    jobId,
    filePath: req.file.path,
    originalFileName: req.file.originalname,
  });

  res.status(202).json({
    jobId,
    status: "queued",
    pollUrl: `/api/jobs/${jobId}`,
  });
});

jobsRouter.get("/meta/queue", (_req, res) => {
  res.json(getQueueSnapshot());
});

jobsRouter.get("/:jobId", (req, res) => {
  const job = getJob(req.params.jobId);
  if (!job) {
    res.status(404).json({ error: "Job not found" });
    return;
  }

  if (job.status === "completed") {
    res.json({
      jobId: job.jobId,
      status: job.status,
      createdAt: job.createdAt,
      updatedAt: job.updatedAt,
      result: job.result,
    });
    return;
  }

  if (job.status === "failed") {
    res.status(500).json({
      jobId: job.jobId,
      status: job.status,
      createdAt: job.createdAt,
      updatedAt: job.updatedAt,
      error: job.error,
    });
    return;
  }

  res.json({
    jobId: job.jobId,
    status: job.status,
    createdAt: job.createdAt,
    updatedAt: job.updatedAt,
  });
});

export { jobsRouter };
