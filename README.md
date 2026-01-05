# GPT-Replica

GPT-Replica is a **minimal GPT-2 style model implementation** built as a learning-focused blueprint inspired by Andrej Karpathy’s  
**“Let’s reproduce GPT-2”** video:  
https://youtu.be/l8pRSuU81PU?si=xeeO8DytpjYXpZNt

This repository aims to closely follow the original GPT-2 architecture while keeping the code simple, readable, and hackable.

## Overview

- Decoder-only Transformer (GPT-2 style)
- Clean, minimal implementation for understanding internals
- Designed as a **blueprint / replica** of GPT-2
- Focused on training-from-scratch concepts

## Training & Performance

- Trained on **NVIDIA H100 GPUs**
- Supports **Distributed Data Parallel (DDP)**
- Includes **hardware-aware optimizations** for efficient large-scale training
- Optimized for throughput and stability on modern accelerators

## Goals

- Learn how GPT-2 works end-to-end
- Provide a solid base for experimentation and extensions
- Keep complexity low while staying faithful to the original design

## Status

🚧 **Work in progress**  
Core architecture, training setup, and usage are complete.  
Fine-tuning code will be added next.
