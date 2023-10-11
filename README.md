# PromptMutant
A implementation of Deepmind's Promptbreeder.
[Promptbreeder: Self-Referential Self-Improvement via Prompt Evoultion](https://arxiv.org/pdf/2309.16797.pdf)

## Installation
```
pip install -e .
```

## Usage
```
promptmutant -prompt "Think out-loud while you answer the question" -dataset "gsm8k" -nPrompts 3 -nMutations 10
```

# Roadmap
- [x] OpenAI LLM Integration
- [x] Cosine Similarity Reward Function
- [x] Zero-order Prompt Generation
- [x] First-order Prompt Generation
- [x] Estimation of Distribution (EDA) Mutation
- [x] EDA Rank and Index Mutation
- [x] Lineage Based Mutation
- [x] Zero-order Hyper-Mutation
- [x] First-order Hyper-Mutation
- [x] Lamarckian Mutation
- [x] Prompt Crossover
- [ ] Context Shuffling
- [ ] Design Additional Reward Functions
- [ ] Ollama Local LLM Integration
