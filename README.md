# Extracting Protein–Protein Interactions from Biomedical Literature with Large Language Models

**Author:** Mariana Oliveira 
**Academic Year:** 2024–2025  
**Master Thesis in Bioinformatics**

---

## Overview

This repository contains all code, prompts, and experimental results associated with my MSc thesis.

> **Goal:**  
> Use Large Language Models (LLMs) to extract *protein–protein interactions (PPIs)* from biomedical research articles, compare them against existing interaction databases (BioGRID, STRING, GENIA), and analyze the effect of model selection and prompt design on extraction quality.

The project evaluates multiple open-source LLMs:

- **BioMistral**
- **Mistral**
- **LLaMA-13B**
- **MedItRON**

---

## Repository Structure

| Folder | Description |
|--------|-------------|
| `Benchmark Databases/` | Contains BioGRID and STRING databases + preprocessing steps to generate benchmark files used to evaluate PPI extractions. |
| `Extraction and Preprocessing/` | Code for PDF → text processing, chunking, and generating embeddings. |
| `PPIs Extractions/` | Code to extract PPIs from 20 research articles using four LLMs (BioMistral, Mistral, LLaMA-13B, MedItRON). |
| `PPIs Extraction Evaluation/` | Contains evaluation code and results comparing extracted PPIs to benchmark datasets (BioGRID, STRING, GENIA). |
| `Testing Prompts/` | Contains code to test 41 prompt variations on five LLMs using a single article. |
| `.gitattributes` | Configuration for Git LFS to track large files. |

