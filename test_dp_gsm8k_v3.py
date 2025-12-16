#!/usr/bin/env python3

import os
import time
import random
import dp_textgrad as tg
from dp_textgrad.tasks import load_task
from dp_textgrad import Variable
from dp_textgrad.dp_es import (
    DPEvolutionStrategy,
    DPScorer,
    DPSelector,
    MutationEngine,
    AdvancedCompositionAccountant,
    DPScorerConfig,
    DPSelectorConfig,
    MutationConfig,
    DPEvolutionConfig,
)
from dp_textgrad.dp_es.population import Candidate

# 1. API Setup
os.environ["OPENAI_API_KEY"] = "sk-ad2c241ee4924e9483cb2059024eda91"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"
tg.set_backward_engine("experimental:openai/deepseek-reasoner", override=True)
engine = tg.get_engine("experimental:openai/deepseek-reasoner")

print("Loading GSM8K_DSPy (20 samples)...")
train_set, val_set, test_set, eval_fn = load_task("GSM8K_DSPy", engine)

# Extract 20 samples for better evaluation
mini_train_data = []
for i in range(min(20, len(train_set))):
    item = train_set.data[i]
    question = item['question']
    answer = item['answer']
    mini_train_data.append((f"Question: {question}", answer))

system_prompt = Variable(
    "You are a helpful assistant. Solve the math problem step by step.",
    role_description="system prompt",
    requires_grad=True
)

# 2. Robust Evaluation Function
def robust_eval_fn(prompt_var):
    total_score = 0
    valid_samples = 0

    for i, (question_str, answer_str) in enumerate(mini_train_data):
        for attempt in range(3):  # Retry up to 3 times
            try:
                # Wrap as Variables
                question_var = Variable(question_str, role_description="question", requires_grad=False)
                answer_var = Variable(answer_str, role_description="answer", requires_grad=False)

                model = tg.BlackboxLLM(engine, system_prompt=prompt_var)
                pred = model(question_var)

                # Evaluation: eval_fn returns "1" for correct, "0" for incorrect
                loss = eval_fn(inputs={"prediction": pred, "ground_truth_answer": answer_var})

                # Convert to score
                loss_value = int(loss.value) if isinstance(loss.value, str) else loss.value
                total_score += float(loss_value)
                valid_samples += 1
                break  # Success, exit retry loop
            except Exception as e:
                print(f"  [Eval Error] Sample {i} Attempt {attempt+1}: {e}")
                time.sleep(1)  # Backoff

    if valid_samples == 0:
        return 0.0

    return total_score / valid_samples

# 3. Strategy-based Mutation
def math_mutation_fn(parent: Candidate, iteration: int, rng: random.Random, feedback):
    base_prompt = parent.variable.get_value()
    strategies = [
        "Add 'Think step by step' and emphasize showing work.",
        "Adopt the persona of a rigorous mathematician.",
        "Ask the model to double check calculations."
    ]
    strategy = strategies[iteration % len(strategies)]

    meta_prompt = f"""Rewrite this system prompt to improve its performance on math reasoning tasks.
    Original: "{base_prompt}"
    Strategy: {strategy}
    Output ONLY the new prompt string."""

    for attempt in range(3):
        try:
            new_val = engine.generate(meta_prompt).strip().strip('"')
            if len(new_val) > 10:
                print(f"  [Mutation] Generated: {new_val[:60]}...")
                return [Variable(new_val, role_description="variant", requires_grad=True)]
        except Exception as e:
            print(f"  [Mutation Error]: {e}")
            time.sleep(1)

    return [Variable(base_prompt, role_description="variant", requires_grad=True)]

# 4. Configure DP Components
scorer_config = DPScorerConfig(
    clipping_value=10.0,
    noise_multiplier=None,
    epsilon_per_candidate=1.0,  # 每个候选的隐私预算
    delta_per_candidate=1e-5,
    composition="advanced",  # 使用高级组合定理
    enable_score_cache=True,
    enable_feedback=False,  # 生产环境禁用反馈
)
scorer = DPScorer(scorer_config)

selector_config = DPSelectorConfig(
    select_k=1,
    epsilon=0.2,
    sensitivity=1.0
)
selector = DPSelector(selector_config)

mutation_config = MutationConfig(
    offspring_per_parent=1,
    allow_identity_offspring=True
)
mutation = MutationEngine(mutation_fn=math_mutation_fn, config=mutation_config)

accountant = AdvancedCompositionAccountant(target_epsilon=10.0, target_delta=1e-4)

evolution_config = DPEvolutionConfig(
    population_size=3,
    parents_to_select=1,
    max_iterations=2,  # Only 2 iterations
    rng_seed=42,
    stop_on_budget=True,
    enable_early_stopping=False,
    enable_elitism=True,
    elite_size=1
)

# 5. Run Optimization
optimizer = DPEvolutionStrategy(
    parameter=system_prompt,
    evaluation_fn=robust_eval_fn,
    scorer=scorer,
    selector=selector,
    mutation_engine=mutation,
    accountant=accountant,
    config=evolution_config
)

print(f"\nInitial Prompt: {system_prompt.get_value()}")
init_score = robust_eval_fn(system_prompt)
print(f"Initial Score: {init_score:.2f}")

print("\nStarting Optimization (Max 2 iterations)...")
optimizer.step()

final_score = robust_eval_fn(system_prompt)
print(f"\nFinal Prompt: {system_prompt.get_value()}")
print(f"Final Score: {final_score:.2f} (Delta: {final_score - init_score:+.2f})")
print(f"Privacy Budget Used: ε={accountant.consumed_epsilon:.4f}")
