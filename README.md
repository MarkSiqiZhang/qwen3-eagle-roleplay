# Joint Alignment Speculative Decoding for Real-Time LLM Role-Play

## Project Overview

**Background & Motivation**
In immersive LLM role-play, the illusion of conversing with a real character relies heavily on the fluidity and speed of the interaction. To truly simulate a live, human-like dialogue, the model must generate text at a pace that matches or exceeds natural human reading and typing speeds. Therefore, maximizing the token generation throughput (Tokens-Per-Second, or TPS) is paramount. When the text generation is sluggish or stutters, it instantly breaks the immersion, constantly reminding the user they are waiting on a machine. However, achieving high sustained throughput with Large Language Models (LLMs) is severely bottlenecked by memory bandwidth during autoregressive decoding, making it incredibly challenging to maintain the dynamic pace required for real-time role-play.

**The Challenge**
Speculative Decoding algorithms (e.g., EAGLE 3) are designed specifically to break this memory bandwidth bottleneck and massively increase throughput. They achieve this by using a lightweight "draft" model to rapidly generate multiple token candidates, which are then verified in parallel by the larger "target" model. 

However, this standard pipeline critically fails in highly stylized role-play tasks. If a target model is explicitly fine-tuned to adopt a specific persona, a standard general-purpose draft model will consistently fail to predict the target's unique stylistic and linguistic distribution. This domain mismatch plummets the token acceptance rate, effectively neutralizing any throughput gains. Conversely, attempting to fine-tune the draft model solely on the target character's sparse dialogue dataset leads to catastrophic overfitting—the draft model loses its fundamental language capabilities before it can learn the persona, resulting in gibberish predictions and degraded inference speed.

**Proposed Solution**
This project introduces Joint Alignment Speculative Decoding. By simultaneously aligning both the Target and Draft models using strategic data mixing and supervised fine-tuning (SFT), we overcome the draft model's "data hunger." This approach bridges the stylistic gap between the two models without destroying the draft model's base capabilities, ultimately restoring the high token acceptance rate and unlocking zero-degradation, high-throughput inference for seamless role-play experiences.


## Objectives & Methodology
**Goal:** Achieve high-speed, real-time role-play inference (high Tokens-Per-Second) without sacrificing the model's persona consistency or instruction-following capabilities.

**Approach:**
1. **Target Model SFT:** Fine-tune a base model (e.g., Llama-3-8B / Qwen-2.5-7B) on high-quality character data.
2. **Draft Model Joint Alignment:** Apply SFT to a lightweight draft model (e.g., 0.5B or an EAGLE draft structure). To prevent overfitting on sparse character data, we employ data mixing strategies (blending general conversational SFT data with character-specific dialogue) and explore lightweight knowledge distillation from the target model.
3. **Speculative Decoding:** Integrate the aligned models to execute fast, parallelized decoding.

---

## Baselines for Comparison
To rigorously evaluate the proposed method, we compare it against three baselines:
* **Baseline 1: Standard Autoregressive Generation.** (Target model only). Maintains persona perfectly but suffers from unacceptable latency.
* **Baseline 2: Standard Speculative Decoding (General Draft).** (Persona-tuned Target + Off-the-shelf Draft). Suffers from massive domain mismatch and high token rejection rates.
* **Baseline 3: Naive Draft Fine-Tuning.** (Draft fine-tuned *only* on small character data). Leads to rapid overfitting and degrades the draft model's fundamental guessing accuracy.

---

## Data Strategy
* **Datasets:** We utilize high-quality, English-only multi-turn datasets such as **RoleBench** and **Character-LLM**. 
* **Character Selection:** The focus is strictly on globally recognized characters (e.g., from the Harry Potter universe). This leverages the models' pre-trained parametric knowledge.
* **Train/Test Split:** We explicitly use the established splits within RoleBench to ensure the model is evaluated on unseen prompts for the *exact same* character it was trained on, guaranteeing a valid benchmark for persona consistency.


---

## Development Roadmap & Course Milestones
- [x] **Milestone 0:** Project conceptualization and initial proposal.
- [ ] **Milestone 1:** Problem definition, baseline setup, and GitHub initialization (Due March 15).
- [ ] **Midterm Presentation:** Presenting goals, problem space, and data strategy (Week 8).
- [ ] **Model Training & Alignment:** Executing Joint Alignment SFT.
- [ ] **Milestone 2:** Finalizing results and evaluation metrics (Due April 26).
- [ ] **Final Deliverable:** Poster, presentation, and WebApp deployment.

