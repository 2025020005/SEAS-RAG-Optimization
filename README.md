# SEAS: Semantic-Enhanced Adaptive Deduplication

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)

OFFICIAL IMPLEMENTATION of the paper: **"SEAS: Semantic-Enhanced Adaptive Deduplication for Efficient Retrieval-Augmented Generation Systems"**.

## 🚀 Overview
Retrieval-Augmented Generation (RAG) systems often suffer from **Context Window Overflow** due to redundant retrieved chunks. **SEAS** is a lightweight, high-throughput deduplication framework designed for real-time RAG pipelines.

### Key Features
- **Hybrid Fingerprinting:** Fuses TF-IDF and MiniLM embeddings to capture both lexical and semantic redundancy.
- **Adaptive Thresholding:** Dynamically adjusts similarity thresholds based on source confidence.
- **Ultra-Low Latency:** Processes 100k documents in <200ms using prefix-based blocking.

## 🛠️ Installation

```bash
git clone [https://github.com/YourUsername/SEAS-RAG.git](https://github.com/YourUsername/SEAS-RAG.git)
cd SEAS-RAG
pip install -r requirements.txt