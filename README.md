# Energy-Based Models from Scratch

## Author's Note

As an engineer who has worked exclusively on early stage startups, I have come to the conclusion that engineering at that stage is, not saying that engineering as a whole isn't, about balancing trade-offs. Assets and resources are scarce for us. I've never had a dedicated DevOps department, nor any particular specialization due to the nature of that environment. Since speed-to-value and customer acquisition are some of the most sought after indicators for companies at that particular stage, I've found myself unable to enjoy treating engineering sometimes as an art. That same nature made it impossible for me to actually deliver on something that I was proud of not only for the customer value it brought, but also from an engineering-as-an-art perspective. And that's ok, work is work, but on my free time I took my time to develop this repository, I do not believe this'll add anything to the base of knowledge in my field, but surely it scratched the need to, pun not-intended, start something from scratch without using the usual third-party libraries that make our lives so much easier at work. I love this, I won't have to maintain it, it's not future proof but rather a proof of the past, a point in which I found myself back to a mathematical root which is tied to what I like about programming, remembering statistical mechanics and re-reading these formulas sure was a plus!

If I may add one last thing, I want to thank Claude Code for writing the rest of this README.md, and also the docstrings for functions, without this amazing tool I wouldn't have been able to spend most of the time actually implementing this stuff, and posting this would've taken much longer.
Hopefully, if you wish to read through this terse code, you'll find it as fun as I did. Thanks, reader!

## AI's README.md

A complete, from-scratch implementation of Energy-Based Models (EBMs) with Langevin dynamics sampling, built using only NumPy. No PyTorch, TensorFlow, or other ML frameworks -- every component from automatic differentiation to the training loop is implemented by hand.

The project includes a practical application: **credit card fraud detection** using EBMs as anomaly detectors.

## What Are Energy-Based Models?

EBMs define a probability distribution over data through an energy function:

```
p(x) = exp(-E(x)) / Z
```

- **E(x)** is a neural network that assigns a scalar energy to each input
- **Z** is the partition function (intractable normalizing constant)
- Low energy = high probability (the model considers x "normal")
- High energy = low probability (the model considers x "anomalous")

Since Z is intractable, EBMs use **contrastive divergence** for training and **Langevin dynamics** for sampling.

## Project Structure

```
ebm/
├── core/                    # Autodiff engine and neural network primitives
│   ├── autodiff.py          # Tensor class with reverse-mode automatic differentiation
│   ├── ops.py               # Differentiable operations (add, matmul, relu, swish, ...)
│   ├── nn.py                # Neural network layers (Linear, ReLU, Swish, Sequential)
│   └── energy.py            # EnergyMLP wrapper with score function computation
├── sampling/                # MCMC sampling from the energy model
│   ├── langevin.py          # Langevin dynamics sampler with gradient clipping & annealing
│   └── replay_buffer.py     # Replay buffer for persistent chain initialization
├── entropy/                 # Entropy estimation for regularization
│   └── knn.py               # k-NN entropy estimator (Kozachenko-Leonenko)
├── training/                # Training loop and optimization
│   ├── trainer.py           # Contrastive divergence training loop
│   └── optimizer.py         # SGD with momentum and AdamW
├── stability/               # Training stability utilities
│   ├── spectral_norm.py     # Spectral normalization via power iteration
│   ├── energy_clamp.py      # Hard and soft energy clamping
│   └── config.py            # Stability diagnostics and monitoring
├── anomaly/                 # Anomaly detection application
│   ├── detector.py          # EBMAnomalyDetector with threshold fitting
│   ├── data.py              # Credit card fraud dataset loader (no pandas)
│   └── evaluate.py          # AUROC, AUPRC, ROC/PR curves, confusion matrix
└── utils/
    └── visualization.py     # Plotting (energy landscapes, histograms, curves)

tests/                       # ~12,900 lines of tests across 12 test modules
demo.ipynb                   # Interactive notebook demonstrating the full pipeline
```

## Key Components

### Automatic Differentiation Engine

The `Tensor` class implements reverse-mode autodiff (backpropagation) from scratch:

```python
from ebm.core import Tensor

a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
c = (a * b).sum()
c.backward()
print(a.grad)  # [4., 5., 6.]
print(b.grad)  # [1., 2., 3.]
```

Supported operations: add, sub, mul, div, matmul, transpose, reshape, sum, mean, relu, sigmoid, softplus, swish, exp, log, pow, sqrt -- all with correct gradient computation and broadcasting support.

### Energy Function

`EnergyMLP` wraps a neural network to output scalar energy values and provides a `score()` method (negative gradient of energy w.r.t. input) used by Langevin dynamics:

```python
from ebm import EnergyMLP

energy_fn = EnergyMLP(input_dim=2, hidden_dims=[128, 128], activation="swish")
E = energy_fn(x)          # Energy values, shape (batch, 1)
s = energy_fn.score(x)    # Score function, shape (batch, 2)
```

### Langevin Dynamics Sampling

Samples from `p(x) = exp(-E(x))/Z` using the discretized overdamped Langevin equation:

```
x_{t+1} = x_t - epsilon * grad_E(x_t) + sqrt(2 * epsilon) * eta,   eta ~ N(0, I)
```

```python
from ebm import langevin_sample

samples = langevin_sample(
    energy_fn, x_init,
    n_steps=100, step_size=0.01,
    noise_scale=1.0, grad_clip=0.03
)
```

Features: per-sample gradient clipping, linear/geometric step size annealing, replay buffer integration, convergence diagnostics.

### Training

Contrastive divergence training with energy and entropy regularization:

```
L = E(x_real).mean() - E(x_fake).mean() + alpha * (E_real^2 + E_fake^2).mean() - lambda * H_knn(x_fake)
```

```python
from ebm import EnergyMLP, AdamW, ReplayBuffer, train

energy_fn = EnergyMLP(input_dim=2, hidden_dims=[128, 128])
optimizer = AdamW(energy_fn.parameters(), lr=1e-4, weight_decay=0.01)
buffer = ReplayBuffer(capacity=10000, sample_dim=2)

history = train(
    energy_fn, optimizer, train_data,
    n_epochs=50, batch_size=128,
    replay_buffer=buffer,
    langevin_steps=40, langevin_step_size=0.01,
    alpha=0.1, lambda_ent=0.01
)
```

### Anomaly Detection

Use trained EBMs for anomaly detection -- anomalies have higher energy than normal data:

```python
from ebm import EBMAnomalyDetector, evaluate_detector

detector = EBMAnomalyDetector(energy_fn)
detector.fit_threshold(X_val, y_val, percentile=95)

predictions = detector.predict(X_test)
result = evaluate_detector(detector, X_test, y_test)
print(f"AUROC: {result.auroc:.4f}, F1: {result.f1:.4f}")
```

### Stability Techniques

- **Spectral normalization**: Constrains layer Lipschitz constants via power iteration
- **Energy clamping**: Hard and soft (tanh-based) clamping to prevent overflow
- **Gradient clipping**: Per-sample clipping during Langevin sampling
- **LR warmup & cosine decay**: Learning rate schedules for stable training
- **Stability diagnostics**: Automatic detection of energy divergence, gradient explosion, mode collapse

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dependencies

- **numpy** -- the only dependency for the core implementation
- **matplotlib** -- visualization (optional, plots degrade gracefully)
- **scipy** -- used only for test validation
- **pytest**, **pytest-cov** -- testing

### Dataset

The credit card fraud detection demo uses the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset. Download `creditcard.csv` and place it in the project root.

## Running Tests

```bash
pytest                          # run all tests
pytest --cov=ebm --cov-report=term-missing   # with coverage
pytest tests/test_autodiff.py   # run a specific module
```

## Demo Notebook

`demo.ipynb` walks through the full pipeline: building an energy function, training with contrastive divergence, sampling via Langevin dynamics, and evaluating anomaly detection on credit card fraud data.

## References

- Yann LeCun et al., "A Tutorial on Energy-Based Learning" (2006)
- Miyato et al., "Spectral Normalization for Generative Adversarial Networks" (2018)
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2017)
- Kozachenko & Leonenko (1987) -- k-NN entropy estimation
- Du & Mordatch, "Implicit Generation and Modeling with Energy-Based Models" (2019)

## License

This project is for educational purposes.
