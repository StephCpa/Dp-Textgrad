import os
import time
import random
import statistics
import numpy as np
import dp_textgrad as tg
from dp_textgrad.tasks import load_task
from dp_textgrad import Variable
from dp_textgrad.dp_es import DPEvolutionStrategy, DPScorer, DPSelector, MutationEngine, AdvancedCompositionAccountant, Candidate, DPScorerConfig, DPSelectorConfig, DPEvolutionConfig

# ==============================================================================
# CONFIGURATION
# ==============================================================================
os.environ["OPENAI_API_KEY"] = "sk-ad2c241ee4924e9483cb2059024eda91"
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"
ENGINE_NAME = "experimental:openai/deepseek-reasoner"

# Medium Scale Settings
BATCH_SIZE = 5         # Increased from 3 to 5 for better granularity
RUNS_PER_LEVEL = 10    # Increased from 2 to 10 for statistical stability
MAX_ITERATIONS = 2     # Keep low for speed
EPSILON_LEVELS = [1.0, 4.0, float('inf')] # Focus on High Privacy, Low Privacy, No Privacy

# ==============================================================================
# SETUP
# ==============================================================================
print(f"Initializing Engine: {ENGINE_NAME}...")
tg.set_backward_engine(ENGINE_NAME, override=True)
engine = tg.get_engine(ENGINE_NAME)

# ==============================================================================
# DATASET
# ==============================================================================
print(f"Initializing GSM8k (Hardcoded Top {BATCH_SIZE} samples)...")

# Hardcoded GSM8k samples (Top 5 from train split)
mini_train = [
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "72"),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "10"),
    ("Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?", "5"),
    ("Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to finish the book tomorrow, how many pages should she read?", "84"),
    ("James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?", "624")
][:BATCH_SIZE]

# Simple evaluation helper
def extract_number(text):
    import re
    # Extract last number in text
    nums = re.findall(r'-?\d+\.?\d*', text)
    return nums[-1] if nums else ""

eval_fn = None # Not used directly, logic moved to robust_eval_fn

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def robust_eval_fn(prompt_var):
    """Evaluate prompt on mini-batch with retries and error handling."""
    total_score = 0
    valid_samples = 0
    
    # Quick string match check to avoid calling API for empty prompts
    if len(prompt_var.get_value().strip()) < 5:
        return 0.0

    for i, (x, y) in enumerate(mini_train):
        for attempt in range(3): # Retry logic
            try:
                model = tg.BlackboxLLM(engine, system_prompt=prompt_var)
                pred = model(x)
                pred_str = pred.get_value()
                
                # Custom robust evaluation logic
                # 1. Try to find the ground truth number exactly in the output
                # 2. Or extract the last number
                
                is_correct = 0.0
                if y in pred_str.split(): # Exact token match
                     is_correct = 1.0
                else:
                    extracted = extract_number(pred_str)
                    if extracted == y:
                        is_correct = 1.0
                
                # DEBUG PRINT
                # if is_correct == 0.0:
                #    print(f"    [Miss] GT: {y} | Pred: {extracted} | Full: {pred_str[:30]}...")
                
                total_score += is_correct
                valid_samples += 1
                break
            except Exception as e:
                # print(f"    [Eval Error] {e}") # Silence errors for cleaner output
                time.sleep(0.5)
    
    if valid_samples == 0:
        return 0.0
        
    return total_score / valid_samples

def math_mutation_fn(parent: Candidate, iteration: int, rng: random.Random, feedback):
    """Specialized mutation for Math tasks."""
    strategies = [
        "Add 'Think step by step' and emphasize showing work.",
        "Adopt the persona of a rigorous mathematician.",
        "Ask the model to double check calculations.",
        "Simplify the instruction to be direct and strict."
    ]
    strategy = strategies[iteration % len(strategies)]
    
    meta_prompt = f"""Rewrite this system prompt to improve its performance on math reasoning tasks.
    Original: "{parent.variable.get_value()}"
    Strategy: {strategy}
    Output ONLY the new prompt string. Do not output quotes."""
    
    for _ in range(2): # Retry mutation
        try:
            val = engine.generate(meta_prompt).strip().strip('"').strip("'")
            if len(val) > 10: 
                return [Variable(val, role_description="variant", requires_grad=True)]
        except: 
            time.sleep(0.5)
            
    # Fallback to parent
    return [Variable(parent.variable.get_value(), role_description="variant", requires_grad=True)]

# ==============================================================================
# MAIN EXPERIMENT LOOP
# ==============================================================================

results = {eps: [] for eps in EPSILON_LEVELS}
initial_prompt_str = "You are a helpful assistant. Solve the math problem step by step."

print("\n" + "="*60)
print(f"ROBUST PRIVACY-UTILITY TRADE-OFF EXPERIMENT")
print(f"Runs: {RUNS_PER_LEVEL} | Batch: {BATCH_SIZE} | Iters: {MAX_ITERATIONS}")
print("="*60)

for eps in EPSILON_LEVELS:
    label = f"ε={eps}" if eps != float('inf') else "Non-DP"
    print(f"\n📊 Testing {label}")
    
    for run in range(RUNS_PER_LEVEL):
        print(f"  Run {run+1}/{RUNS_PER_LEVEL}...", end="", flush=True)
        start_t = time.time()
        
        # 1. Reset Prompt
        system_prompt = Variable(initial_prompt_str, role_description="system prompt", requires_grad=True)
        
        # 2. Configure DP Parameters
        if eps == float('inf'):
            # Non-DP: Infinite budget, No noise
            scorer_config = DPScorerConfig(clipping_value=1e9, noise_multiplier=0.0, epsilon_per_candidate=100, delta_per_candidate=0, enable_score_cache=True, composition="basic")
            selector_config = DPSelectorConfig(select_k=1, epsilon=100, sensitivity=1.0)
            accountant = AdvancedCompositionAccountant(target_epsilon=float('inf'), target_delta=1.0)
        else:
            # DP: Distribute budget.
            # We have MAX_ITERATIONS steps. Per step cost = eps / (iters + buffer)
            # Allocate more to Scorer (quality) than Selector.
            total_steps = MAX_ITERATIONS
            budget_per_step = eps / (total_steps + 0.5)

            scorer_eps = budget_per_step * 0.8
            selector_eps = budget_per_step * 0.2

            scorer_config = DPScorerConfig(clipping_value=10.0, epsilon_per_candidate=scorer_eps, delta_per_candidate=1e-5, enable_score_cache=True, composition="advanced")
            selector_config = DPSelectorConfig(select_k=1, epsilon=selector_eps, sensitivity=1.0)
            accountant = AdvancedCompositionAccountant(target_epsilon=eps + 0.1, target_delta=1e-4)

        # 3. Initialize Optimizer
        optimizer = DPEvolutionStrategy(
            parameter=system_prompt,
            evaluation_fn=robust_eval_fn,
            scorer=DPScorer(scorer_config),
            selector=DPSelector(selector_config),
            mutation_engine=MutationEngine(mutation_fn=math_mutation_fn),
            accountant=accountant,
            config=DPEvolutionConfig(
                population_size=3, 
                parents_to_select=1,
                max_iterations=MAX_ITERATIONS, 
                enable_early_stopping=False
            )
        )
        
        # 4. Run Step
        try:
            optimizer.step()
            final_score = robust_eval_fn(system_prompt)
            results[eps].append(final_score)
            print(f" Score: {final_score:.2f} ({time.time()-start_t:.1f}s)")
        except Exception as e:
            print(f" Failed ({str(e)[:50]}...). Score: 0.00")
            results[eps].append(0.0)

# ==============================================================================
# REPORT
# ==============================================================================
print("\n" + "="*60)
print("FINAL ROBUST REPORT")
print("="*60)
print(f"{'Epsilon':<10} | {'Mean Score':<12} | {'Std Dev':<10} | {'Min':<6} | {'Max':<6}")
print("-" * 60)

for eps in EPSILON_LEVELS:
    s = results[eps]
    label = str(eps) if eps != float('inf') else "Inf"
    mean_s = np.mean(s) if s else 0.0
    std_s = np.std(s) if s else 0.0
    min_s = min(s) if s else 0.0
    max_s = max(s) if s else 0.0
    
    print(f"{label:<10} | {mean_s:.4f}       | {std_s:.4f}     | {min_s:.2f}   | {max_s:.2f}")

print("-" * 60)
print("Raw Data:", results)
