# LLM-From-Scratch
Here is my implementation of a GPT-2 124M-style large language model, modeled after the book Build a Large Language Model (From Scratch) by Sebastian Raschka. The implementation includes causal multi-head self-attention, token embeddings, transformer blocks, and autoregressive text generation. 

The objective of this project was to develop a foundational understanding of large language model architectures. 

Each transformer block consists of:
* Multi-head self-attention
* Layer normalization
* Residual connections
* Feedforward neural network
* GELU activation
* Dropout regularization

During each training iteration, the model produces logits over the vocabulary for each token position of a batch of token sequences. Cross-entropy loss is calculated, and gradients are computed using backpropagation. AdamW updates the model parameters. The training and validation losses are periodically evaluated, along with a sample text generated after each epoch to inspect model behavior.

The model uses a custom-built autoregressive text generation pipeline to generate tokens sequentially. The current implementation generates 50 additional tokens per sample and uses greedy decoding rather than sampling.

The dataset used, MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x, is a reasoning dataset; however, the implementation does not pre-process the data in a way that would train the model for reasoning, as dataset messages are concatenated without preserving explicit conversation roles. This would be something to update in future iterations. 

Additional future improvements include:
* Implementing temperature, top-p, and top-k sampling
* Introducing learning-rate scheduling and gradient clipping
* Adding checkpoint resumption
