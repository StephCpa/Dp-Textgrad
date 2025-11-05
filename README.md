![Logo](assets/logo_full.png)

<!--- BADGES: START --->
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zou-group/TextGrad/blob/main/examples/notebooks/Prompt-Optimization.ipynb)
[![GitHub license](https://img.shields.io/badge/License-MIT-blue.svg)][#license-gh-package]
[![Nature](https://img.shields.io/badge/Nature-Paper-green?logo=nature&logoColor=white)](https://www.nature.com/articles/s41586-025-08661-4)
[![Arxiv](https://img.shields.io/badge/arXiv-2406.07496-B31B1B.svg)][#arxiv-paper-package]
[![Documentation Status](https://readthedocs.org/projects/dp-textgrad/badge/?version=latest)][#docs-package]
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dp-textgrad)][#pypi-package]
[![PyPI](https://img.shields.io/pypi/v/dp-textgrad)][#pypi-package]
[![Conda - Platform](https://img.shields.io/conda/pn/conda-forge/dp-textgrad?logo=anaconda&style=flat)][#conda-forge-package]
[![Conda (channel only)](https://img.shields.io/conda/vn/conda-forge/dp-textgrad?logo=anaconda&style=flat&color=orange)][#conda-forge-package]

[#license-gh-package]: https://lbesson.mit-license.org/
[#arxiv-paper-package]: https://arxiv.org/abs/2406.07496
[#docs-package]: https://dp_textgrad.readthedocs.io/en/latest/?badge=latest
[#pypi-package]: https://pypi.org/project/dp-textgrad/
[#conda-forge-package]: https://anaconda.org/conda-forge/dp-textgrad
[#nature-link]: https://www.nature.com/articles/s41586-025-08661-4

<!--- BADGES: END --->

## TextGrad: Automatic ''Differentiation'' via Text

An autograd engine -- for textual gradients!

TextGrad is a powerful framework  building automatic ``differentiation'' via text.
TextGrad implements backpropagation through text feedback provided by LLMs, strongly building on the gradient metaphor

We provide a simple and intuitive API that allows you to define your own loss functions and optimize them using text feedback.
This API is similar to the Pytorch API, making it simple to adapt to your usecases.

![Analogy with Torch](assets/analogy.png)


### Updates:

**19th March 2025**

TextGrad published in [Nature](https://www.nature.com/articles/s41586-025-08661-4)!

**Past Updates**:

We are introducing a new engine based on [litellm](https://github.com/BerriAI/litellm). This should allow
you to use any model you like, as long as it is supported by litellm. This means that now
**Bedrock, Together, Gemini and even more** are all supported by TextGrad!

This should be seen as experimental but we plan to depreciate the old engines in the future.

In addition to this, with the new engines it should be easy to enable and disable caching.  

We are in the process of testing these new engines and deprecating the old engines. If you have any issues, please let us know!

The new litellm engines can be loaded with the following code: 

An example of loading a litellm engine:
```python
engine = get_engine("experimental:gpt-4o", cache=False)

# this also works with

set_backward_engine("experimental:gpt-4o", cache=False)
```

Be sure to set the relevant environment variables for the new engines!

An example of forward pass:
```python

import httpx
from dp_textgrad.engine_experimental.litellm import LiteLLMEngine

LiteLLMEngine("gpt-4o", cache=True).generate(content="hello, what's 3+4", system_prompt="you are an assistant")

image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"
image_data = httpx.get(image_url).content

LiteLLMEngine("gpt-4o", cache=True).generate(content=[image_data, "what is this my boy"], system_prompt="you are an assistant")
```

In the examples folder you will find two new notebooks that show how to use the new engines.


## QuickStart
If you know PyTorch, you know 80% of TextGrad.
Let's walk through the key components with a simple example. Say we want to use GPT-4o to solve a simple
reasoning problem.

The question is *If it takes 1 hour to dry 25 shirts under the sun, how long will it take to dry 30 shirts under the sun? Reason step by step.* (Thanks, [Reddit User](https://www.reddit.com/r/OpenAI/comments/18q479x/comment/kf444es/))

```python
import dp_textgrad as tg

tg.set_backward_engine("gpt-4o", override=True)

# Step 1: Get an initial response from an LLM.
model = tg.BlackboxLLM("gpt-4o")
question_string = ("If it takes 1 hour to dry 25 shirts under the sun, "
                   "how long will it take to dry 30 shirts under the sun? "
                   "Reason step by step")

question = tg.Variable(question_string,
                       role_description="question to the LLM",
                       requires_grad=False)

answer = model(question)
```

> :warning: **answer: To determine how long it will take to dry 30 shirts under the sun,**
> **we can use a proportional relationship based on the given information.**
> **Here’s the step-by-step reasoning: [.....]**
> **So, it will take 1.2 hours (or 1 hour and 12 minutes) to dry 30 shirts under the sun.**


As you can see, **the model's answer is incorrect.** We can optimize the answer using TextGrad to get the correct answer.

```python

answer.set_role_description("concise and accurate answer to the question")

# Step 2: Define the loss function and the optimizer, just like in PyTorch!
# Here, we don't have SGD, but we have TGD (Textual Gradient Descent)
# that works with "textual gradients".
optimizer = tg.TGD(parameters=[answer])
evaluation_instruction = (f"Here's a question: {question_string}. "
                           "Evaluate any given answer to this question, "
                           "be smart, logical, and very critical. "
                           "Just provide concise feedback.")


# TextLoss is a natural-language specified loss function that describes
# how we want to evaluate the reasoning.
loss_fn = tg.TextLoss(evaluation_instruction)
```
> :brain: **loss: [...] Your step-by-step reasoning is clear and logical,**
> **but it contains a critical flaw in the assumption that drying time is**
> **directly proportional to the number of shirts. [...]**

```python
# Step 3: Do the loss computation, backward pass, and update the punchline.
# Exact same syntax as PyTorch!
loss = loss_fn(answer)
loss.backward()
optimizer.step()
answer
```

> :white_check_mark: **answer: It will still take 1 hour to dry 30 shirts under the sun,**
> **assuming they are all laid out properly to receive equal sunlight.**

We have many more examples around how TextGrad can optimize all kinds of variables -- code, solutions to problems, molecules, prompts, and all that!

### Tutorials

We have prepared a couple of tutorials to get you started with TextGrad. The order of this
tutorial is what we would recommend to follow for a beginner. You can run them directly in Google Colab by clicking on the links below (but
you need an OpenAI/Anthropic key to run the LLMs).

<div align="center">

| Tutorial                                           | Difficulty                                                      | Colab Link                                                                                                                                                                                                    |
|----------------------------------------------------|-----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1. Introduction to TextGrad Primitives             | ![](https://img.shields.io/badge/Level-Beginner-green.svg)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zou-group/TextGrad/blob/main/examples/notebooks/Tutorial-Primitives.ipynb)              |
| 2. Solution Optimization                           | ![](https://img.shields.io/badge/Level-Beginner-green.svg)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zou-group/TextGrad/blob/main/examples/notebooks/Tutorial-Solution-Optimization.ipynb)   |
| 3. Optimizing a Code Snippet and Define a New Loss | ![](https://img.shields.io/badge/Level-Beginner-green.svg)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zou-group/dp-textgrad/blob/main/examples/notebooks/Tutorial-Test-Time-Loss-for-Code.ipynb) |
| 4. Prompt Optimization                             | ![](https://img.shields.io/badge/Level-Intermediate-yellow.svg) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zou-group/TextGrad/blob/main/examples/notebooks/Tutorial-Prompt-Optimization.ipynb)     |
| 5. MultiModal Optimization                         | ![](https://img.shields.io/badge/Level-Beginner-green.svg)      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zou-group/TextGrad/blob/main/examples/notebooks/Tutorial-MultiModal.ipynb)              |

</div>



### Installation

You can install TextGrad using any of the following methods.

**With `pip`**:

```bash
pip install dp-textgrad
```

**With `conda`**:

```sh
conda install -c conda-forge dp-textgrad
```

> :bulb: The conda-forge package for `dp-textgrad` is maintained [here](https://github.com/conda-forge/dp-textgrad-feedstock).

**Bleeding edge installation with `pip`**:

```sh
pip install git+https://github.com/zou-group/dp-textgrad.git
```

**Installing dp-textgrad with vllm**:

```sh
pip install dp-textgrad[vllm]
```

See [here](https://pip.pypa.io/en/stable/cli/pip_install/) for more details on various methods of pip installation.

### Differential Privacy Evolution Strategy (DP-ES)

TextGrad now supports **differential privacy-aware optimization** through DP-ES! Optimize text variables while providing provable privacy guarantees.

#### 🚀 Quick Start with Preset Configurations

```python
import dp_textgrad as tg
from dp_textgrad import Variable, get_dp_config, DPEvolutionStrategy
from dp_textgrad.dp_es import DPScorer, DPSelector, MutationEngine, AdvancedCompositionAccountant

# Load a preset privacy configuration
config = tg.get_dp_config("medium_privacy")  # or "high_privacy", "low_privacy"

# Define your task
prompt = Variable("Solve this.", role_description="instruction", requires_grad=True)

def evaluate(var):
    # Your evaluation function using private data
    return some_score

# Create DP components from config
scorer = DPScorer(config.scorer)
selector = DPSelector(config.selector)
mutation_engine = MutationEngine(mutation_fn=your_mutation_fn, config=config.mutation)
accountant = AdvancedCompositionAccountant(
    target_epsilon=config.target_epsilon,
    target_delta=config.target_delta
)

# Run DP-ES optimization
optimizer = DPEvolutionStrategy(
    parameter=prompt,
    evaluation_fn=evaluate,
    scorer=scorer,
    selector=selector,
    mutation_engine=mutation_engine,
    accountant=accountant,
    config=config.evolution
)
optimizer.step()

# Check privacy consumption
print(f"Privacy used: ε={accountant.consumed_epsilon}, δ={accountant.consumed_delta}")
```

#### 📚 Tutorials & Examples
- **Basic Tutorial**: [Tutorial-DP-Evolution-Strategy.ipynb](examples/notebooks/Tutorial-DP-Evolution-Strategy.ipynb) - Complete introduction to DP-ES
- **Comparison**: [Tutorial-DP-ES-Comparison.ipynb](examples/notebooks/Tutorial-DP-ES-Comparison.ipynb) - DP-ES vs standard TGD
- **Quick Start Script**: [dp_es_quickstart.py](examples/dp_es_quickstart.py) - Minimal working example
- **Design Document**: [DP-TextGrad via DP-ES.md](DP‑TextGrad%20via%20DP‑ES.md) - Architecture and research background

#### 🔧 Configuration Presets

View all available privacy configurations:
```python
tg.print_config_comparison()
```

| Privacy Level | Target ε | Use Case | Description |
|--------------|----------|----------|-------------|
| `high_privacy` | ε ≤ 1.0 | Healthcare, Finance | Strong privacy for sensitive PII |
| `medium_privacy` | 1.0 < ε ≤ 3.0 | General Business | Balanced privacy-performance |
| `low_privacy` | 3.0 < ε ≤ 6.0 | Aggregated Data | Better performance, reasonable privacy |
| `minimal_privacy` | ε > 6.0 | Testing, Public Data | Weak privacy, focus on performance |

#### 🔑 Key Components
- **`DPScorer`**: Evaluation with clipping and noise injection
- **`DPSelector`**: Privacy-preserving parent selection (Noisy Top-k, Exponential Mechanism)
- **`MutationEngine`**: Generate candidate variations with optional critique pipeline
- **`PrivacyAccountant`** / **`AdvancedCompositionAccountant`**: Track privacy budget (ε, δ)
- **`DPEvolutionStrategy`**: Main optimization controller
- **`CritiquePipeline`**: DP-aware critique generation and selection
- **`FeedbackSanitiser`**: Prevent privacy leakage in textual feedback

#### ⚙️ Environment & Testing
- **Python version**: >=3.9
- **Core dependencies**: Listed in `requirements.txt` (openai, httpx, diskcache, etc.)
- **Running tests**: `pytest` from repository root
- **Data requirements**: Provide your own task evaluation functions

#### 🔬 Privacy Verification

Empirically verify that DP-ES provides the promised privacy guarantees:

```bash
# Quick sanity check (~30 seconds)
python evaluation/quick_privacy_check.py

# Comprehensive verification (~5-10 minutes)
python evaluation/example_privacy_verification.py

# Generate visual reports
python evaluation/privacy_visualization.py
```

**Verification includes:**
- ✅ Neighboring database tests (core DP property)
- ✅ Membership inference attack resistance
- ✅ Noise distribution validation
- ✅ Privacy budget accuracy checks

See [evaluation/README_PRIVACY_VERIFICATION.md](evaluation/README_PRIVACY_VERIFICATION.md) for details.

#### 📖 Documentation & Technical Deep Dives

**Getting Started:**
- 🚀 [Quick Reference Guide](docs/QUICK_REFERENCE.md) - One-page visual summary of mutation algorithms, DP mechanisms, formulas, and best practices
- 📘 [Basic Tutorials](examples/notebooks/) - Jupyter notebooks for hands-on learning
- ⚡ [Quick Start Script](examples/dp_es_quickstart.py) - Minimal working example with preset configs

**Advanced Topics:**
- 🧬 [Mutation Strategies Guide](dp-textgrad/dp_es/MUTATION_STRATEGIES.md) - Complete guide to intelligent mutation algorithms (LLM-guided, adaptive, gradient-guided, crossover)
- 🔄 [Mutation Upgrade Guide](MUTATION_UPGRADE_GUIDE.md) - How to upgrade from simple string concatenation to intelligent mutations (2-3x performance boost)
- 🔬 [Technical Deep Dive](docs/DP_ES_TECHNICAL_EXPLANATION.md) - Comprehensive explanation of mutation algorithms and differential privacy implementation with numerical examples
- 🎮 [Interactive Demo](examples/interactive_dp_explanation.py) - Step-by-step walkthrough of mutation evolution and DP protection

**Key Concepts:**
- **Intelligent Mutations**: Use LLMs to understand semantics and generate meaningful variations instead of simple string concatenation
- **Adaptive Strategy**: Automatically adjusts exploration/exploitation balance during optimization (recommended for most use cases)
- **DP Guarantees**: Score clipping → Gaussian noise injection → Privacy budget tracking with advanced composition
- **Privacy-Performance Trade-off**: Larger ε = better performance but weaker privacy; smaller ε = stronger privacy but slower convergence

## More detailed examples

### Minimal Instance Optimization Example

TextGrad can optimize unstructured variables, such as text. Let us have an initial solution to a math problem that we want to improve. Here's how to do it with TextGrad, using GPT-4o:

```python
tg.set_backward_engine("gpt-4o")

initial_solution = """To solve the equation 3x^2 - 7x + 2 = 0, we use the quadratic formula:
x = (-b ± √(b^2 - 4ac)) / 2a
a = 3, b = -7, c = 2
x = (7 ± √((-7)^2 - 4 * 3(2))) / 6
x = (7 ± √(7^3) / 6
The solutions are:
x1 = (7 + √73)
x2 = (7 - √73)"""

# Define the variable to optimize, let requires_grad=True to enable gradient computation
solution = tg.Variable(initial_solution,
                       requires_grad=True,
                       role_description="solution to the math question")

# Define the optimizer, let the optimizer know which variables to optimize, and run the loss function

loss_fn = tg.TextLoss("You will evaluate a solution to a math question. Do not attempt to solve it yourself, do not give a solution, only identify errors. Be super concise.")

optimizer = tg.TGD(parameters=[solution])
loss = loss_fn(solution)
```

Output:
> Variable(value=Errors:
> 1. Incorrect sign in the discriminant calculation: it should be b^2 - 4ac, not b^2 + 4ac.
> 2. Incorrect simplification of the quadratic formula: the denominator should be 2a, not 6.
> 3. Final solutions are missing the division by 2a., role=response from the language model, grads=)

```python
loss.backward()
optimizer.step()
print(solution.value)
```

Output:
> To solve the equation 3x^2 - 7x + 2 = 0, we use the quadratic formula:
> x = (-b ± √(b^2 - 4ac)) / 2a
>
> Given:
> a = 3, b = -7, c = 2
>
> Substitute the values into the formula:
> x = (7 ± √((-7)^2 - 4(3)(2))) / 6
> x = (7 ± √(49 - 24)) / 6
> x = (7 ± √25) / 6
> x = (7 ± 5) / 6
>
> The solutions are:
> x1 = (7 + 5) / 6 = 12 / 6 = 2
> x2 = (7 - 5) / 6 = 2 / 6 = 1/3

### Minimal Prompt Optimization Example
TextGrad can also optimize prompts in PyTorch style! Here's how to do it with TextGrad, using GPT-4o for feedback, and optimizing a prompt for gpt-3.5-turbo:
```python
import dp_textgrad as tg
from dp_textgrad.tasks import load_task

llm_engine = tg.get_engine("gpt-3.5-turbo")
tg.set_backward_engine("gpt-4o")

_, val_set, _, eval_fn = load_task("BBH_object_counting", llm_engine)
question_str, answer_str = val_set[0]
question = tg.Variable(question_str, role_description="question to the LLM", requires_grad=False)
answer = tg.Variable(answer_str, role_description="answer to the question", requires_grad=False)
```

Question:
> I have two stalks of celery, two garlics, a potato, three heads of broccoli, a carrot, and a yam. How many vegetables do I have?

Ground Truth Answer:
> 10

```python
system_prompt = tg.Variable("You are a concise LLM. Think step by step.",
                            requires_grad=True,
                            role_description="system prompt to guide the LLM's reasoning strategy for accurate responses")

model = tg.BlackboxLLM(llm_engine, system_prompt=system_prompt)
optimizer = tg.TGD(parameters=list(model.parameters()))

prediction = model(question)
```

Prediction:
> You have a total of seven vegetables: two stalks of celery, two garlics, one potato, three heads of broccoli, one carrot, and one yam.

```python
loss = eval_fn(inputs=dict(prediction=prediction, ground_truth_answer=answer))
```

Loss denoting accuracy:
> Variable(value=0, grads=)

```python
loss.backward()
```

System prompt gradients:
> ... 2. **Encourage Explicit Summation**: - The prompt should encourage the model to explicitly state the summation process. This can help in verifying the accuracy of the count. For example, "Explain your calculations clearly and verify the total."....

```python
optimizer.step()
```

New system prompt value:
> You are a concise LLM. Think step by step. Prioritize accuracy in your calculations. Identify and count each item individually. Explain your calculations clearly and verify the total. After calculating, review your steps to ensure the total is correct. If you notice a discrepancy in your count, re-evaluate the list and correct any mistakes.

```python
prediction = model(question)
```

New prediction:
> Let's count the number of each vegetable:
> 1. Celery stalks: 2
> 2. Garlics: 2
> 3. Potato: 1
> 4. Broccoli heads: 3
> 5. Carrot: 1
> 6. Yam: 1
>
> Now, let's add up the total number of vegetables:
> 2 + 2 + 1 + 3 + 1 + 1 = 10
>
> You have a total of 10 vegetables.

## Resources

### Inspiration
Many existing works greatly inspired this project! Here is a non-exhaustive list:
- 📚 [PyTorch](https://github.com/pytorch/pytorch/) The one and only. We owe a ton to PyTorch, hard to do justice here.
- 📚 [DSPy](https://github.com/stanfordnlp/dspy) is a pioneer in writing LM-based programs in many different ways! Has been a huge inspiration for us.
- 📚 [Micrograd](https://github.com/karpathy/micrograd): A tiny autograd engine greatly inspired our simple design!
- 📚 [ProTeGi](https://github.com/microsoft/LMOps/tree/main/prompt_optimization): We owe the term "Textual Gradients" to ProTeGi!
- 📚 [Reflexion](https://github.com/noahshinn/reflexion): A self-reflection that showed us the power of text-based reflection!

### Citation
```bibtex
@article{yuksekgonul2025optimizing,
  title={Optimizing generative AI by backpropagating language model feedback},
  author={Yuksekgonul, Mert and Bianchi, Federico and Boen, Joseph and Liu, Sheng and Lu, Pan and Huang, Zhi and Guestrin, Carlos and Zou, James},
  journal={Nature},
  volume={639},
  pages={609--616},
  year={2025},
}
```


### Contributors

We are grateful for all the help we got from our contributors!

<!-- readme: contributors -start -->
<table>
	<tbody>
		<tr>
            <td align="center">
                <a href="https://github.com/vinid">
                    <img src="https://avatars.githubusercontent.com/u/2234699?v=4" width="100;" alt="vinid"/>
                    <br />
                    <sub><b>Federico Bianchi</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/mertyg">
                    <img src="https://avatars.githubusercontent.com/u/29640736?v=4" width="100;" alt="mertyg"/>
                    <br />
                    <sub><b>Mert Yuksekgonul</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/nihalnayak">
                    <img src="https://avatars.githubusercontent.com/u/5679782?v=4" width="100;" alt="nihalnayak"/>
                    <br />
                    <sub><b>Nihal Nayak</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/sugatoray">
                    <img src="https://avatars.githubusercontent.com/u/10201242?v=4" width="100;" alt="sugatoray"/>
                    <br />
                    <sub><b>Sugato Ray</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/lupantech">
                    <img src="https://avatars.githubusercontent.com/u/17663606?v=4" width="100;" alt="lupantech"/>
                    <br />
                    <sub><b>Pan Lu</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/ruanwz">
                    <img src="https://avatars.githubusercontent.com/u/4874?v=4" width="100;" alt="ruanwz"/>
                    <br />
                    <sub><b>David Ruan</b></sub>
                </a>
            </td>
		</tr>
		<tr>
            <td align="center">
                <a href="https://github.com/sanowl">
                    <img src="https://avatars.githubusercontent.com/u/99511815?v=4" width="100;" alt="sanowl"/>
                    <br />
                    <sub><b>San </b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/huangzhii">
                    <img src="https://avatars.githubusercontent.com/u/10902358?v=4" width="100;" alt="huangzhii"/>
                    <br />
                    <sub><b>Zhi Huang</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/ajms">
                    <img src="https://avatars.githubusercontent.com/u/2394425?v=4" width="100;" alt="ajms"/>
                    <br />
                    <sub><b>Albert</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/tboen1">
                    <img src="https://avatars.githubusercontent.com/u/46495807?v=4" width="100;" alt="tboen1"/>
                    <br />
                    <sub><b>tboen1</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/AtakanTekparmak">
                    <img src="https://avatars.githubusercontent.com/u/59488384?v=4" width="100;" alt="AtakanTekparmak"/>
                    <br />
                    <sub><b>Atakan Tekparmak</b></sub>
                </a>
            </td>
		</tr>
	<tbody>
</table>
<!-- readme: contributors -end -->
