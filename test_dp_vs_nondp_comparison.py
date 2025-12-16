#!/usr/bin/env python3
"""
DP-ES vs non-DP Baselines: Comprehensive Comparison

Comparing:
1. DP-ES: The proposed method (Evolution Strategy + Differential Privacy)
2. Non-DP ES: The ablation baseline (Evolution Strategy only, no privacy noise)
3. Non-DP TGD: The standard baseline (Textual Gradient Descent)

Hypothesis:
Non-DP ES > DP-ES > Non-DP TGD
- Non-DP ES should be best (same powerful algorithm, no noise).
- DP-ES should be close to Non-DP ES (showing low privacy cost).
- Both ES methods should beat TGD (showing algorithm superiority).
"""

import os
import time
import statistics
import random
import numpy as np
from typing import List, Dict, Any, Callable

os.environ["OPENAI_API_KEY"] = "sk-ccai3pDr1nrkZOs_CJUs4w"
os.environ["OPENAI_API_BASE"] = "https://llmapi.paratera.com"

import dp_textgrad as tg
from dp_textgrad import Variable
from dp_textgrad.optimizer import TextualGradientDescent
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

print("=" * 80)
print(" DP-ES vs Non-DP ES vs Non-DP TGD: Three-Way Comparison")
print("=" * 80)

# Configure Engine
tg.set_backward_engine("experimental:openai/DeepSeek-V3.2-Exp", override=True)
llm_engine = tg.get_engine("experimental:openai/DeepSeek-V3.2-Exp")

# Shared Experiment Parameters
INITIAL_PROMPT = "写一个问候语"
MAX_ITERATIONS = 5
NUM_RUNS = 3

print(f"Task: Optimize prompt '{INITIAL_PROMPT}'")
print(f"Settings: {MAX_ITERATIONS} iterations, {NUM_RUNS} runs per method")

# ============================================================================
# Evaluation Function
# ============================================================================
def evaluation_fn(var: Variable) -> float:
    """Unified evaluation function."""
    prompt = var.get_value()
    score = 0.0

    # Length check (0-3)
    length = len(prompt)
    if 20 < length < 100:
        score += 3.0
    elif length < 20:
        score += 1.0
    else:
        score += 2.0

    # Keyword coverage (0-5)
    keywords = ["友好", "具体", "正式", "场合", "适合", "要求", "清晰", "目标", "对象"]
    keyword_count = sum(1 for kw in keywords if kw in prompt)
    score += min(keyword_count, 5)

    # Uniqueness (0 to -2 penalty)
    words = prompt.split()
    if not words: return 0.0
    unique_ratio = len(set(words)) / len(words)
    if unique_ratio < 0.7: score -= 2.0
    elif unique_ratio < 0.85: score -= 1.0

    return max(score, 0.0)

initial_score = evaluation_fn(Variable(INITIAL_PROMPT, role_description="test", requires_grad=False))
print(f"Initial Score: {initial_score:.2f}")

# ============================================================================
# Helper: Common Mutation Function
# ============================================================================
def common_mutation_fn(parent: Candidate, iteration: int, rng: random.Random, feedback):
    """LLM-driven mutation for both DP-ES and Non-DP ES."""
    prompt = f"""You are a prompt optimizer. Rewrite the following prompt to make it better, more specific, and professional.
    Original: "{parent.variable.get_value()}"
    Generate 2 variations. Output ONLY the new prompts, one per line."""
    
    try:
        response = llm_engine.generate(prompt)
        lines = response.split('\n')
        variations = []
        for line in lines:
            # Fix: correctly stripped quotes
            clean = line.strip().strip('"').strip("'").strip('- ').strip()
            if len(clean) > 5:
                variations.append(Variable(clean, role_description="variant", requires_grad=True))
        
        # Fallback if parsing fails
        if not variations:
             variations = [
                 Variable(f"{parent.variable.get_value()} (improved)", role_description="variant", requires_grad=True),
                 Variable(f"{parent.variable.get_value()} (refined)", role_description="variant", requires_grad=True)
             ]
        return variations[:2]
    except Exception as e:
        return [Variable(parent.variable.get_value(), role_description="variant", requires_grad=True)]

# ============================================================================
# Method 1: DP-ES (The Proposal)
# ============================================================================
def run_dp_es(run_id: int) -> Dict:
    print(f"\n[DP-ES] Run {run_id+1}")
    target = Variable(INITIAL_PROMPT, role_description="prompt", requires_grad=True)
    
    # DP Config (Moderate Privacy)
    scorer = DPScorer(DPScorerConfig(
        clipping_value=10.0,
        epsilon=0.5,
        delta=1e-5,
        enable_score_cache=True
    ))
    selector = DPSelector(DPSelectorConfig(select_k=1, epsilon=0.1, sensitivity=1.0))
    mutation = MutationEngine(mutation_fn=common_mutation_fn)
    accountant = AdvancedCompositionAccountant(target_epsilon=8.0, target_delta=1e-4)
    
    optimizer = DPEvolutionStrategy(
        parameter=target,
        evaluation_fn=evaluation_fn,
        scorer=scorer,
        selector=selector,
        mutation_engine=mutation,
        accountant=accountant,
        config=DPEvolutionConfig(
            population_size=4,
            max_iterations=MAX_ITERATIONS,
            enable_early_stopping=True,
            enable_elitism=True
        )
    )
    
    start = time.time()
    optimizer.step()
    elapsed = time.time() - start
    
    return {
        "final_score": evaluation_fn(target),
        "improvement": evaluation_fn(target) - initial_score,
        "time": elapsed,
        "epsilon": accountant.consumed_epsilon,
        "history": optimizer.get_optimization_stats().get('score_history', [])
    }

# ============================================================================
# Method 2: Non-DP ES (The Ablation Baseline)
# ============================================================================
def run_nondp_es(run_id: int) -> Dict:
    print(f"\n[Non-DP ES] Run {run_id+1}")
    target = Variable(INITIAL_PROMPT, role_description="prompt", requires_grad=True)
    
    # Effective Non-DP Config: Infinite budget, No noise
    scorer = DPScorer(DPScorerConfig(
        clipping_value=1e9,   # No clipping
        noise_multiplier=0.0, # No noise
        epsilon=100.0,        # Dummy epsilon
        delta=0.0,
        enable_score_cache=True
    ))
    selector = DPSelector(DPSelectorConfig(select_k=1, epsilon=100.0, sensitivity=1.0))
    mutation = MutationEngine(mutation_fn=common_mutation_fn)
    accountant = AdvancedCompositionAccountant(target_epsilon=float('inf'), target_delta=1.0)
    
    optimizer = DPEvolutionStrategy(
        parameter=target,
        evaluation_fn=evaluation_fn,
        scorer=scorer,
        selector=selector,
        mutation_engine=mutation,
        accountant=accountant,
        config=DPEvolutionConfig(
            population_size=4,
            max_iterations=MAX_ITERATIONS,
            enable_early_stopping=True,
            enable_elitism=True
        )
    )
    
    start = time.time()
    optimizer.step()
    elapsed = time.time() - start
    
    return {
        "final_score": evaluation_fn(target),
        "improvement": evaluation_fn(target) - initial_score,
        "time": elapsed,
        "epsilon": 0.0,
        "history": optimizer.get_optimization_stats().get('score_history', [])
    }

# ============================================================================
# Method 3: Non-DP TGD (The Standard Baseline)
# ============================================================================
def run_nondp_tgd(run_id: int) -> Dict:
    print(f"\n[Non-DP TGD] Run {run_id+1}")
    target = Variable(INITIAL_PROMPT, role_description="prompt", requires_grad=True)
    optimizer = TextualGradientDescent(parameters=[target], engine=llm_engine)
    
    history = []
    start = time.time()
    
    for _ in range(MAX_ITERATIONS):
        score = evaluation_fn(target)
        history.append(score)
        
        # Backward pass
        loss = Variable(f"Score: {score}. Improve specificness and tone.", role_description="feedback")
        # Fix: Accept any arguments to be robust against API changes
        loss.set_grad_fn(lambda *args, **kwargs: "Make it more professional and specific.")
        loss.backward()
        optimizer.step()
    
    elapsed = time.time() - start
    
    return {
        "final_score": evaluation_fn(target),
        "improvement": evaluation_fn(target) - initial_score,
        "time": elapsed,
        "epsilon": 0.0,
        "history": history
    }

# ============================================================================
# Main Execution Loop
# ============================================================================
methods = {
    "DP-ES": run_dp_es,
    "Non-DP ES": run_nondp_es,
    "Non-DP TGD": run_nondp_tgd
}

results = {name: [] for name in methods}

for run in range(NUM_RUNS):
    print(f"\n--- Batch {run+1} ---")
    for name, func in methods.items():
        try:
            res = func(run)
            results[name].append(res)
            print(f"  -> Score: {res['final_score']:.2f} (+{res['improvement']:.2f})")
        except Exception as e:
            print(f"  -> Failed: {e}")

# ============================================================================
# Final Report
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS SUMMARY")
print("="*80)
print(f"{ 'Method':<15} | {'Improvement':<15} | {'Time (s)':<10} | {'Epsilon':<10}")
print("-" * 60)

stats = {}
for name, res_list in results.items():
    if not res_list: continue
    imps = [r['improvement'] for r in res_list]
    times = [r['time'] for r in res_list]
    eps = [r['epsilon'] for r in res_list]
    
    stats[name] = np.mean(imps)
    print(f"{name:<15} | {np.mean(imps):.2f} ± {np.std(imps):.2f}  | {np.mean(times):.1f}       | {np.mean(eps):.2f}")

print("-" * 60)

# Hypothesis Check
es_gap = stats.get("Non-DP ES", 0) - stats.get("Non-DP TGD", 0)
privacy_cost = stats.get("Non-DP ES", 0) - stats.get("DP-ES", 0)

print("\nAnalysis:")
print(f"1. Algorithm Benefit (ES vs TGD): {es_gap:+.2f} points")
if es_gap > 0:
    print("   ✅ Evolution Strategy outperforms Gradient Descent.")
else:
    print("   ⚠️  Gradient Descent held its ground.")

print(f"2. Privacy Cost (Non-DP ES vs DP-ES): {privacy_cost:+.2f} points")
if privacy_cost < 1.0:
    print("   ✅ Low privacy cost (High Utility).")
else:
    print("   ⚠️  High privacy cost.")