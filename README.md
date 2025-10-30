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

mariana-olivetree/
│
├── Benchmark Databases/
│ Contains BioGRID and STRING databases, plus preprocessing steps to generate benchmark files used for evaluating the extracted PPIs.
│
├── Extraction and Preprocessing/
│ Contains processing scripts used to convert PDF articles into text chunks and generate their embeddings.
│
├── PPIs Extractions/
│ Contains the code used to extract PPIs from 20 research articles using four LLMs (BioMistral, Mistral, LLaMA-13B, MedItRON).
│
├── PPIs Extraction Evaluation/
│ Contains the code and results of the evaluation of extracted PPIs against benchmark datasets (BioGRID, STRING, GENIA) for the four models.
│
├── Testing Prompts/
│ Contains code used to test 41 prompt variations on five LLMs using a single article as a controlled prompt-engineering experiment.
│
└── .gitattributes
