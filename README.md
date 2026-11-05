# Simple Anomaly Generator

A lightweight, programmatic 2D image generator built to create controlled, ground-truth datasets for benchmarking computer vision models. 

## The Problem
Standard anomaly-detection benchmark datasets often contain inherent label inaccuracies or structural inconsistencies. It is difficult to rigorously test a model's architectural stability or perform ablation studies when the ground-truth data itself is flawed.

## The Solution
This script generates 2D arrays (visually appearing as grayscale squares with controlled "stains" or anomalies) where the ground-truth is absolute. By parametrically controlling the noise, standard deviation, and structural boundaries (rigid vs. fuzzy) of these anomalies, researchers can isolate and benchmark how models handle label noise and feature similarity without the bottleneck of cleaning real-world datasets.

## Features
* **Baseline Noise Simulation:** Generates parameterized background matrices.
* **Controlled Anomalies:** Injects target anomalies with strict mathematical constraints.
* **Structural Distortion:** Configurable boundary scattering and fuzzy edge generation for edge-case testing.
