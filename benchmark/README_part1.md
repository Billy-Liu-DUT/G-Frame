# Chemistry Performance Evaluation Benchmarks

This repository contains a suite of benchmarks designed to evaluate the performance of Large Language Models (LLMs) on a variety of chemistry-related tasks. The benchmarks cover long-form explanatory answers, and multiple-choice questions from various chemistry domains.

## Benchmarks Included

1.  [**ChemJudge**](#1-chemjudge) - Evaluates long-form, expert-level chemical knowledge using an "LLM as a Judge" approach.
2.  [**ThChem1.0**](#2-thchem10) - A multiple-choice question set focusing on fundamental chemistry concepts.
3.  [**ThChem2.0**](#3-thchem20) - An advanced multiple-choice set where some questions may not have a correct answer.

---

### 1. ChemJudge

**ChemJudge** is designed to assess a model's ability to generate deep, accurate, and well-structured explanations on complex, graduate-level chemistry topics.

#### Evaluation Model

This benchmark operates on an **"LLM as a Judge"** framework. The questions are open-ended and require detailed, expert-level responses. The quality of the generated answers is intended to be evaluated by another advanced LLM (the "judge"), which scores the response based on accuracy, depth, clarity, and completeness.

#### Data Format

The dataset is a JSON array of objects, where each object represents a single evaluation prompt.

-   `theme`: A string indicating the specific topic or sub-discipline of chemistry.
-   `question`: A string containing the detailed, open-ended question for the LLM to answer.

**Example:**
```json
{
  "theme": "Wittig Reaction Mechanism & Selectivity",
  "question": "Explain the mechanism of the Wittig reaction, including ylide formation and oxaphosphetane intermediate decomposition, and discuss how reaction conditions and substrate structure influence E/Z selectivity."
}
```

---

### 2. ThChem1.0

**ThChem1.0** is a multiple-choice question (MCQ) benchmark designed to test an LLM's foundational knowledge across various chemistry topics. Each question has a single correct answer.

#### Data Format

The dataset is a JSON array of objects, with each object containing an instruction, a question with options, and the correct output.

-   `instruction`: A string containing the task description, which is consistent across all entries.
-   `input`: A string that includes the `Question:` and the list of `Options:`.
-   `output`: A string representing the letter of the correct option (e.g., "A", "B", "C").

**Example:**
```json
{
  "instruction": "Which of the following options is correct?",
  "input": "Question: Has the largest atomic radius.\nOptions:\nA. Berylium\nB. Boron\nC. Carbon\nD. Oxygen\nE. Fluorine ",
  "output": "A"
}
```

---

### 3. ThChem2.0

**ThChem2.0** is an advanced multiple-choice question (MCQ) benchmark that introduces an additional layer of difficulty: some questions may not have a correct answer among the provided options. This tests the model's ability to identify incorrect or flawed premises, rather than simply selecting the "best" fit.

#### Data Format

The format is similar to ThChem1.0, but with a key difference in the expected output for questions with no valid answer.

-   `instruction`: A string containing the task description, which explicitly states that some questions may not have a correct answer.
-   `input`: A string that includes the `Question:` and the list of `Options:`.
-   `output`: A string representing the letter of the correct option, OR the string `"null"` if no option is correct.

**Example (with a correct answer):**
```json
{
  "instruction": "Your task is to complete the following multiple-choice questions...",
  "input": "Question: In the ground state, has two electrons in one (and only one) of the p orbitals.\nOptions:\nA. Berylium\nB. Boron\nC. Carbon\nD. Oxygen ",
  "output": "D"
}
```

**Example (with no correct answer):**
```json
{
  "instruction": "Your task is to complete the following multiple-choice questions...",
  "input": "Question: The half-life of 3 H is about 12 years. How much of a 4mg sample will remain after 36 years?\nOptions:\nA. 0.25mg\nB. 1mg\nC. 2mg\nD. 4mg ",
  "output": "null"
}
```