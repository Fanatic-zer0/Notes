# Important 9 concepts of AI..

---

## 1. Tokenization

**What it is:**  
Neural networks do not work on raw text; they work on numbers. **Tokenization** is the process of turning text into a sequence of discrete symbols (tokens), and then mapping each token to an integer ID.

### Why tokenization exists

- Text is variable length; neural nets need fixed-size numeric inputs.
- Languages have huge vocabularies, spelling variants, typos, and rare words.
- Tokenization lets models:
  - Compress many possible strings into a manageable vocabulary
  - Share structure between related words (walk, walks, walking)
  - Handle arbitrary text, including code and emojis

### Common tokenization strategies

1. **Word-level**  
   - Each word is a token.  
   - Simple but:
     - Huge vocabulary (hundreds of thousands of words)
     - Fails badly on out-of-vocabulary (OOV) words and typos

2. **Character-level**  
   - Each character is a token.  
   - Vocabulary is tiny, but sequences become very long.
   - Harder for the model to learn long-range structure.

3. **Subword-level (BPE, SentencePiece, Unigram) — what modern LLMs use**  
   - Start from smaller units (bytes, characters).
   - Repeatedly merge the most frequent adjacent pairs (Byte Pair Encoding/BPE).
   - Over time, common chunks like “ing”, “tion”, “pre”, “##ing” become individual tokens.
   - Example:  
     - “walking” → `["walk", "ing"]`  
     - “tokenization” → `["token", "ization"]`

### Why tokenization matters in practice

- **Context length & cost:** Commercial APIs typically bill per token, not per character. “Tokens” are roughly 3–4 characters of English on average.
- **Multilingual performance:** Good tokenization reduces fragmentation in languages with complex morphology (e.g., agglutinative languages).
- **Quality of generation:** Better tokenization preserves useful patterns (e.g., morphemes, code constructs) that models can learn.

---

## 2. Text Decoding

**What it is:**  
Once the model has processed input tokens, it outputs a **probability distribution over the vocabulary for the next token**. **Decoding** is how you turn that distribution into actual text.

### The basic loop

1. Input tokens → model → probability distribution over possible next tokens.
2. A **decoding algorithm** picks the next token.
3. Append that token and repeat until:
   - an end-of-sequence token,
   - a length limit, or
   - some other stopping criteria.

### Key decoding strategies

1. **Greedy decoding**  
   - Always pick the most probable next token.  
   - Pros:
     - Fast
     - Deterministic, good for tasks like classification, extraction, function calling.
   - Cons:
     - Boring, repetitive outputs
     - Can get stuck in loops (“the the the the…”)

2. **Sampling-based methods**  
   These add controlled randomness for creativity.

   - **Temperature sampling**
     - Scales logits before softmax.
     - Temperature < 1 → more deterministic (peaky).
     - Temperature > 1 → more random, creative.

   - **Top-k sampling**
     - Keep only the k most likely tokens, renormalize, and sample from them.

   - **Top-p (nucleus) sampling**  
     - Take the smallest set of tokens whose cumulative probability ≥ p (e.g., 0.9).
     - Sample from that dynamic set.
     - Adapts to distribution shape better than fixed k.

3. **Beam search**
   - Maintains multiple candidate sequences (“beams”) in parallel.
   - Selects sequences that maximize overall probability.
   - Popular in translation, but used less in pure LLM chat settings.

### Why decoding matters

- **Same model, different behavior**: Changing only decoding can make a model look “creative” or “rigid.”
- **Task fit**:
  - Deterministic (greedy/low-temp) for code, tools, structured output.
  - Stochastic (top-p/high-temp) for brainstorming, writing, ideation.
- **Safety and reliability**: Certain decoding strategies can make hallucinations more or less likely.

---

## 3. Prompt Engineering

**What it is:**  
**Prompt engineering** is the craft of writing inputs that shape model behavior **without changing its weights**.

### Why it works

LLMs are pattern matchers over text. The way you phrase the task, examples, and constraints creates patterns the model tries to continue. Good prompts give it:

- A clear objective
- Constraints and style
- Examples to imitate
- Intermediate reasoning steps to follow

### Core components of a strong prompt

- **Role / persona**: “You are a security engineer…”
- **Task**: “Generate a runbook for…”
- **Constraints**: “Answer in JSON”, “Be concise”, “Don’t fabricate citations.”
- **Context**: Logs, docs, prior messages, etc.
- **Examples (few-shot)**: Demonstrations of input → output format.

### Techniques often used

1. **Few-shot prompting**  
   - You provide a handful of exemplars:
     - Input A → Output A  
     - Input B → Output B  
     - Now Input C → ?
   - The model imitates style, reasoning, and format.

2. **Chain-of-thought prompting (CoT)**  
   - Ask the model to reason step-by-step:
     - “Think step by step.”
     - “Show your intermediate reasoning before the final answer.”
   - Especially useful for:
     - Math
     - Logic puzzles
     - Multi-step coding tasks

### Why prompt engineering is widely used

- **Fast iteration** vs training or fine-tuning.
- **Cheap**: Only costs tokens, not GPU time for training.
- **Non-invasive**: No model ownership required; works with closed APIs.

Over time, prompt engineering techniques often get “baked into” system prompts, orchestrators, or even model training data.

---

## 4. Multi-step AI Agents

**What they are:**  
A plain LLM just **maps text → text**. An **agent** wraps an LLM in a loop with:

- **Tools** (APIs, search, code execution, DBs)
- **Memory**
- **Planning logic**

So instead of just answering once, the LLM **decides what to do next**, calls tools, observes results, and repeats until done.

### Core loop (mental model)

1. **Perceive**: Read current user goal + context + memory.
2. **Plan**: Decide on next subtask.
3. **Act**: Call a tool (search, run code, query DB, etc.).
4. **Observe**: Read tool output.
5. **Update plan**: Decide next step or finish.

This continues until:

- The goal is reached.
- A budget limit (tokens, tool calls, time) is hit.
- The agent decides no further progress is possible.

### Typical components

- **Planner**: Breaks the high-level goal into steps.
- **Tool-caller**: Structured calls to external APIs.
- **Memory**: Keeps track of previous steps, results, user preferences.
- **Controller**: Enforces loops, timeouts, safety constraints.

### Why agents matter

- Enable **real workflows**: booking, monitoring, automation, data pipelines.
- Bridge between **static LLMs** and **dynamic environments**.
- Turn LLMs into something closer to “autonomous” systems (within guardrails).

Failure modes: tool misuse, infinite loops, hallucinated tools, or fragile planning. Production systems require strict safety, constraints, and observability—very similar to orchestrating microservices from an SRE perspective.

---

## 5. Retrieval-Augmented Generation (RAG)

**What it is:**  
A vanilla LLM answers from what is stored in its **weights**, which are frozen after training. **RAG** pairs an LLM with a **retrieval system** connected to an external knowledge base.

### High-level flow

1. **User query** comes in.
2. A **retriever** searches a knowledge store (docs, PDFs, DB, wiki, etc.) for relevant passages.
3. The **retrieved chunks** plus the user query are fed to the LLM.
4. The LLM writes an answer using that retrieved context.

This **grounds** the answer in external data rather than just model memory.

### Why RAG is powerful

- Handles **fresh information** (new policies, recent events, company data).
- Reduces hallucinations (if retrieval is good and system is designed properly).
- Avoids expensive full-model retraining for every knowledge update.

### Typical architecture

- **Ingestion pipeline**:
  - Chunk documents.
  - Embed each chunk into a vector.
  - Store in a vector database (e.g., FAISS, Milvus, pgvector, etc.).

- **Query pipeline**:
  - Embed the user’s question.
  - Perform vector search to retrieve top-k chunks.
  - Optionally, re-rank with a secondary model.
  - Concatenate retrieved chunks with the question in the prompt.

### Design levers

- Chunk size and overlap.
- Retrieval strategy (vector, BM25, hybrid).
- Reranking and query rewriting.
- Prompt structure (how context is presented and constrained).

---

## 6. Reinforcement Learning from Human Feedback (RLHF)

**What it is:**  
**RLHF** is a reinforcement learning technique used to **align** models with human preferences—helpful, safe, and clear behavior rather than just “statistically likely” text.

### RLHF pipeline (simplified)

1. **Base model pretraining**  
   - LLM is trained on massive unlabeled text via next-token prediction.

2. **Supervised fine-tuning (often)**  
   - Human-labeled input → output pairs teach desired behaviors directly.

3. **Reward model training**  
   - Collect data:
     - For a given prompt, generate multiple candidate responses.
     - Human annotators choose which response they prefer.
   - Train a **reward model** to predict which answer humans would prefer.
   - The reward model becomes a **proxy for human judgment**.

4. **Reinforcement learning step**  
   - Use RL algorithms (e.g., PPO) to adjust the base LLM so it tends to:
     - Generate responses that the reward model scores highly.
     - Avoid low-scoring behaviors (unhelpful, unsafe, incoherent).

### Why RLHF matters

- Major reason chat-style models feel:
  - Polite
  - On-topic
  - Safety-conscious
- Allows “steering” model behavior at scale without labeling every token.

### Limitations and risks

- **Reward hacking**: Model finds ways to satisfy the reward model without truly doing what humans want.
- Over-optimization can reduce diversity, creativity, and truthfulness if the reward model is flawed.
- Requires lots of human annotation (expensive, tricky to scale to all domains).

---

## 7. Variational Autoencoders (VAE)

**What they are:**  
A **variational autoencoder** is a **generative model** that learns a probability distribution over data by encoding it into a **latent space** and decoding from that space.

### Core structure

- **Encoder network**:
  - Maps input \(x\) (image, audio, etc.) to parameters of a latent distribution, typically a mean \(\mu\) and variance \(\sigma^2\).

- **Latent space**:
  - A low-dimensional continuous space \(z\) sampled from \(N(\mu, \sigma^2)\).

- **Decoder network**:
  - Takes \(z\) and reconstructs \(\hat{x}\).

The training objective balances:

1. **Reconstruction loss**:
   - Make \(\hat{x}\) close to \(x\).

2. **Regularization (KL divergence)**:
   - Make the inferred latent distribution close to a prior (often standard normal).

### Generative usage

Once trained:

- Sample a new latent vector \(z\) from the prior (e.g., standard normal).
- Feed \(z\) into the decoder.
- Output is a newly generated sample that looks like training data.

### Role in modern systems (e.g., image/video generators)

- VAEs are often used as a **latent compressor**:
  - Instead of running diffusion directly in pixel space, encode images/videos into a smaller latent space.
  - Generative models (like diffusion) operate on this compact representation.
  - Decoder (VAE) maps latent back to full-resolution pixels.

- Benefits:
  - Massive efficiency gains.
  - Easier learning in lower-dimensional representation.

---

## 8. Diffusion Models

**What they are:**  
Diffusion models generate data by learning to **reverse a noise process**.

### Training process (forward + reverse)

1. **Forward process (noising)**:
   - Start from a real sample \(x_0\) (e.g., an image).
   - Gradually add noise across many time steps to get \(x_1, x_2, \dots, x_T\) where \(x_T\) is almost pure noise.
   - This process is usually fixed and known.

2. **Reverse process (denoising)**:
   - Train a neural network to predict the noise (or the clean sample) at each time step:
     - Given a noisy sample \(x_t\), time index \(t\), and optional conditioning (like text),
     - Predict either the added noise or the original data.
   - The model effectively learns how to **denoise step-by-step**.

### Inference / generation

- Start from pure noise \(x_T\).
- Iteratively apply the learned denoising step:
  - \(x_T \rightarrow x_{T-1} \rightarrow \dots \rightarrow x_0\)
- If conditioned on text, the model generates samples matching that description.

### Why diffusion works well

- Stable and scalable training, often with better diversity vs older GANs.
- Natural fit with **latent spaces** (via VAEs), enabling high-res image and video generation.
- Text-to-image, image-to-image, inpainting, and text-to-video all use diffusion as a backbone.

---

## 9. Low-Rank Adaptation (LoRA)

**What it is:**  
**LoRA** is an efficient fine-tuning method that adapts a large model **without updating all its parameters**.

### Core idea

- Consider a dense layer with weight matrix \(W \in \mathbb{R}^{d \times k}\).
- Standard fine-tuning: update all entries of \(W\).
- LoRA:
  - Keep \(W\) **frozen**.
  - Add two small trainable matrices \(A \in \mathbb{R}^{d \times r}\) and \(B \in \mathbb{R}^{r \times k}\), where \(r\) (rank) is small.
  - Effective weight becomes:
    \[
    W' = W + A B
    \]
  - Only \(A\) and \(B\) are trained.

### Why this helps

- **Huge parameter savings**:
  - Instead of updating billions of parameters, you train a much smaller number.
- **Modularity**:
  - Different LoRA adapters for different domains or tasks.
  - Can be swapped, stacked, or merged without retraining the base model.
- **Practical deployment**:
  - Keep one base model, load task-specific LoRAs as needed.
  - Great for on-device or resource-constrained setups.

### Use cases

- Domain specialization:
  - Medical, legal, financial, gaming, etc.
- Style personalization:
  - Specific voice, tone, or character behavior.
- Rapid experiments:
  - Try many adapters without retraining full models.

---

## Summarizing it all together

- **Tokenization**: How text becomes numbers.
- **Decoding**: How probabilities become actual words.
- **Prompt engineering**: How to steer frozen models with clever inputs.
- **Agents**: How LLMs interact with tools and act over multiple steps.
- **RAG**: How models stay up to date and grounded in external knowledge.
- **RLHF**: How models are aligned with human preferences at scale.
- **VAE**: How complex data gets compressed into useful latent spaces.
- **Diffusion**: How noise is turned into images, audio, or video.
- **LoRA**: How huge models are efficiently adapted to niche tasks.


