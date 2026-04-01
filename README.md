# Joint Alignment Speculative Decoding for Real-Time LLM Role-Play

## Abstract

Real-time LLM role-play requires a model to generate responses that are not only coherent and instruction-following, but also highly consistent with a predefined character persona and fast enough to sustain natural interaction. This requirement is especially important in applications such as game NPC dialogue, interactive storytelling, and conversational character agents, where long response delays can easily disrupt immersion. Although speculative decoding has emerged as an effective method for accelerating large language model inference by using a lightweight draft model to propose candidate tokens for parallel verification by a larger target model, its effectiveness drops substantially in stylized role-play settings. Once the target model is fine-tuned on persona-specific dialogue, its token distribution shifts toward character-dependent language patterns, tone, and expression style. A general-purpose draft model cannot accurately approximate this shifted distribution, which leads to low token acceptance rates and weak throughput gains. Meanwhile, training or adapting the draft model using only such narrow data can lead to overfitting and may weaken its general conversational ability. Therefore, instead of training a new draft model from scratch, this project proposes to adapt the draft model through continued pretraining on a mixture of general conversational data and persona-specific dialogue data.

To address this problem, this project proposes **Joint Alignment Speculative Decoding**, a framework for improving persona-conditioned generation while preserving the efficiency benefits of speculative decoding. The framework consists of three components: persona-specific supervised fine-tuning of the target model, draft model adaptation through continued-pretraining-inspired data mixing, and integration of the two models within a speculative decoding pipeline. Its main modeling contribution is a draft adaptation strategy in which large-scale general conversational data, such as UltraChat 200K, is mixed with character dialogue data, such as RoleBench or Character-LLM, so that the draft model can gradually shift toward persona-relevant stylistic distributions without losing its general predictive ability.

---

## Contributions

This project makes three contributions:

1. **Draft Adaptation Strategy.** We investigate a continued-pretraining-inspired strategy for adapting the draft model in persona-conditioned speculative decoding, aiming to reduce draft-target distribution mismatch while preserving general predictive ability.

2. **Empirical Validation.** We produce an adapted model whose performance is empirically validated on selected persona-based role-play tasks, with systematic comparisons against autoregressive, general-draft, and naive-finetuning baselines.

3. **Interactive Frontend.** We develop a lightweight interactive frontend that enables direct testing of the model and provides a practical demonstration of its behavior in real-time role-play interaction.

---

## Methodology

### 1. Target Model Persona SFT

We fine-tune a base language model (Qwen3-4B) on high-quality character dialogue data from RoleBench. This step shifts the target model's token distribution toward character-specific language patterns, personality traits, and expression style.

### 2. Draft Model Adaptation via Data Mixing

Training a draft model purely on character dialogue data leads to severe overfitting due to limited dataset size. To mitigate this, we adopt a data mixing strategy inspired by continual pretraining.

We construct a mixed training dataset consisting of:
- **General conversational data** (e.g., UltraChat 200K) to retain broad language modeling capability
- **Character dialogue data** (e.g., RoleBench) to learn persona-specific stylistic patterns

This allows the draft model to gradually shift toward persona-relevant distributions without losing its general predictive ability, significantly improving the token acceptance rate during speculative decoding.

### 3. Speculative Decoding Integration

We integrate the aligned draft and target models into an EAGLE 3 speculative decoding pipeline. The draft model generates multiple token candidates via tree-based beam search, and the target model verifies them in parallel. Because both models are aligned through joint training, the draft model can better approximate the target model's predictions, leading to higher acceptance rates and improved tokens-per-second (TPS).

---

## Baselines

We compare the proposed method against three baselines:

| Baseline | Setup | Expected Issue |
|---|---|---|
| **B1: Autoregressive** | Target model only, standard decoding | Strong persona, but low TPS (~19 tok/s) |
| **B2: General Draft** | Persona-tuned target + off-the-shelf EAGLE 3 draft | Distribution mismatch, low acceptance rate |
| **B3: Naive Draft SFT** | Draft fine-tuned only on character data | Catastrophic overfitting, degraded predictions |

---

## Data Strategy

- **Character Data:** RoleBench (English), focusing on globally recognized characters (e.g., Jack Sparrow) to leverage the model's pre-trained parametric knowledge
- **General Data:** UltraChat 200K for draft model data mixing
- **Train/Test Split:** We use the established instruction-generalization splits within RoleBench to ensure the model is evaluated on unseen prompts for the same character

---

## Final Deliverables

1. **Trained Models:** Persona-aligned target and draft models for Jack Sparrow role-play
2. **Reproducible Pipeline:** End-to-end scripts for data preprocessing, model training, evaluation, and inference
3. **Interactive Web Demo:** A lightweight frontend (Gradio) for real-time role-play interaction, demonstrating side-by-side comparison of autoregressive vs. speculative decoding with live TPS display
4. **Evaluation Report:** Systematic benchmarks comparing TPS, token acceptance rate, and persona consistency across all baselines and the proposed method

---

## Development Roadmap

- [x] **Milestone 0:** Project conceptualization and initial proposal
- [x] **Milestone 1:** Problem definition, baseline setup, and GitHub initialization
- [ ] **Midterm Presentation:** Goals, problem space, and data strategy
- [ ] **Model Training & Alignment:** Target SFT + Draft adaptation via data mixing
- [ ] **Milestone 2:** Finalizing results and evaluation metrics (Due April 26)
- [ ] **Final Deliverable:** Poster, presentation, and web demo deployment
