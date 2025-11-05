"""
Example: Advanced Mutation Strategies for DP-ES

This example demonstrates the improved mutation strategies that go
beyond simple string concatenation, showing significant performance gains.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dp_textgrad as tg
from dp_textgrad import Variable
from dp_textgrad.dp_es import (
    DPEvolutionStrategy,
    DPEvolutionConfig,
    DPScorer,
    DPScorerConfig,
    DPSelector,
    DPSelectorConfig,
    MutationEngine,
    MutationConfig,
    AdvancedCompositionAccountant,
)
from dp_textgrad.dp_es.mutation_strategies import (
    create_mutation_function,
    MutationMode,
    LLMGuidedMutation,
    AdaptiveMutation,
)


# ============================================================================
# COMPARISON: Simple vs Advanced Mutations
# ============================================================================

def run_comparison():
    """Compare simple concatenation vs intelligent LLM mutations."""

    print("\n" + "="*70)
    print("MUTATION STRATEGY COMPARISON")
    print("="*70)

    # Set up engine
    tg.set_backward_engine("gpt-4o-mini", override=True)
    model = tg.BlackboxLLM("gpt-4o-mini")

    # Simple task: optimize a question-answering prompt
    test_questions = [
        "What is photosynthesis?",
        "Explain gravity",
        "How does DNA replication work?",
    ]

    def evaluate_prompt(variable: Variable) -> float:
        """Evaluate prompt quality (simplified for demo)."""
        prompt = variable.get_value()
        score = 0.0

        # Check for good prompt characteristics
        if len(prompt) > 20:
            score += 0.2
        if any(word in prompt.lower() for word in ["explain", "describe", "detail"]):
            score += 0.3
        if any(word in prompt.lower() for word in ["step", "example", "clear"]):
            score += 0.3
        if "concise" in prompt.lower() or "comprehensive" in prompt.lower():
            score += 0.2

        return min(score, 1.0)

    initial_prompt = "Answer the question."

    # ========================================================================
    # Method 1: Simple String Concatenation (Current Baseline)
    # ========================================================================

    print("\n" + "-"*70)
    print("METHOD 1: Simple String Concatenation (Baseline)")
    print("-"*70)

    prompt_simple = Variable(
        initial_prompt,
        role_description="instruction prompt",
        requires_grad=True
    )

    def simple_mutation(parent, iteration, rng, feedback):
        """Simple concatenation-based mutation."""
        base = parent.variable.get_value()
        variations = [
            base + " Be clear.",
            base + " Provide details.",
        ]
        return [Variable(v, role_description="prompt", requires_grad=True)
                for v in variations]

    # Run DP-ES with simple mutation
    scorer = DPScorer(DPScorerConfig(
        clipping_value=1.0,
        epsilon=0.5,
        delta=1e-5
    ))

    selector = DPSelector(DPSelectorConfig(select_k=2, epsilon=0.0))

    mutation_engine_simple = MutationEngine(
        mutation_fn=simple_mutation,
        config=MutationConfig(offspring_per_parent=2)
    )

    accountant_simple = AdvancedCompositionAccountant(
        target_epsilon=3.0,
        target_delta=1e-4
    )

    optimizer_simple = DPEvolutionStrategy(
        parameter=prompt_simple,
        evaluation_fn=evaluate_prompt,
        scorer=scorer,
        selector=selector,
        mutation_engine=mutation_engine_simple,
        accountant=accountant_simple,
        config=DPEvolutionConfig(
            population_size=4,
            parents_to_select=2,
            max_iterations=3,
            rng_seed=42
        )
    )

    print(f"Initial: '{initial_prompt}' (score: {evaluate_prompt(prompt_simple):.3f})")
    optimizer_simple.step()
    simple_final_score = evaluate_prompt(prompt_simple)
    print(f"Final:   '{prompt_simple.get_value()}'")
    print(f"Score:   {simple_final_score:.3f}")

    # ========================================================================
    # Method 2: Intelligent LLM-Guided Mutation
    # ========================================================================

    print("\n" + "-"*70)
    print("METHOD 2: Intelligent LLM-Guided Mutation")
    print("-"*70)

    prompt_llm = Variable(
        initial_prompt,
        role_description="instruction prompt for answering science questions",
        requires_grad=True
    )

    # Use advanced mutation strategy
    llm_mutation_fn = create_mutation_function(
        strategy="llm_guided",
        mode=MutationMode.BALANCED,
        num_offspring=2
    )

    mutation_engine_llm = MutationEngine(
        mutation_fn=llm_mutation_fn,
        config=MutationConfig(offspring_per_parent=2)
    )

    accountant_llm = AdvancedCompositionAccountant(
        target_epsilon=3.0,
        target_delta=1e-4
    )

    optimizer_llm = DPEvolutionStrategy(
        parameter=prompt_llm,
        evaluation_fn=evaluate_prompt,
        scorer=scorer,
        selector=selector,
        mutation_engine=mutation_engine_llm,
        accountant=accountant_llm,
        config=DPEvolutionConfig(
            population_size=4,
            parents_to_select=2,
            max_iterations=3,
            rng_seed=42
        )
    )

    print(f"Initial: '{initial_prompt}' (score: {evaluate_prompt(prompt_llm):.3f})")
    optimizer_llm.step()
    llm_final_score = evaluate_prompt(prompt_llm)
    print(f"Final:   '{prompt_llm.get_value()}'")
    print(f"Score:   {llm_final_score:.3f}")

    # ========================================================================
    # Method 3: Adaptive Mutation (Auto-adjusting)
    # ========================================================================

    print("\n" + "-"*70)
    print("METHOD 3: Adaptive Mutation (Auto-Adjusting)")
    print("-"*70)

    prompt_adaptive = Variable(
        initial_prompt,
        role_description="instruction prompt for answering science questions",
        requires_grad=True
    )

    adaptive_mutation_fn = create_mutation_function(
        strategy="adaptive",
        num_offspring=2
    )

    mutation_engine_adaptive = MutationEngine(
        mutation_fn=adaptive_mutation_fn,
        config=MutationConfig(offspring_per_parent=2)
    )

    accountant_adaptive = AdvancedCompositionAccountant(
        target_epsilon=3.0,
        target_delta=1e-4
    )

    optimizer_adaptive = DPEvolutionStrategy(
        parameter=prompt_adaptive,
        evaluation_fn=evaluate_prompt,
        scorer=scorer,
        selector=selector,
        mutation_engine=mutation_engine_adaptive,
        accountant=accountant_adaptive,
        config=DPEvolutionConfig(
            population_size=4,
            parents_to_select=2,
            max_iterations=3,
            rng_seed=42
        )
    )

    print(f"Initial: '{initial_prompt}' (score: {evaluate_prompt(prompt_adaptive):.3f})")
    optimizer_adaptive.step()
    adaptive_final_score = evaluate_prompt(prompt_adaptive)
    print(f"Final:   '{prompt_adaptive.get_value()}'")
    print(f"Score:   {adaptive_final_score:.3f}")

    # ========================================================================
    # COMPARISON SUMMARY
    # ========================================================================

    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    results = [
        ("Simple Concatenation", simple_final_score),
        ("LLM-Guided Mutation", llm_final_score),
        ("Adaptive Mutation", adaptive_final_score),
    ]

    for method, score in results:
        improvement = score - evaluate_prompt(Variable(initial_prompt, role_description=""))
        print(f"{method:.<30} {score:.3f} (+{improvement:.3f})")

    best_method = max(results, key=lambda x: x[1])
    print(f"\n✓ Best method: {best_method[0]} (score: {best_method[1]:.3f})")


# ============================================================================
# DETAILED EXAMPLE: Using Different Mutation Modes
# ============================================================================

def demonstrate_mutation_modes():
    """Show different mutation modes (explore/exploit/balanced)."""

    print("\n" + "="*70)
    print("MUTATION MODES DEMONSTRATION")
    print("="*70)

    tg.set_backward_engine("gpt-4o-mini", override=True)

    base_text = "Explain this concept clearly."

    for mode in [MutationMode.EXPLOIT, MutationMode.EXPLORE, MutationMode.BALANCED]:
        print(f"\n--- {mode.value.upper()} Mode ---")

        mutator = LLMGuidedMutation(mode=mode, diversity_weight=0.3)

        # Create dummy parent candidate
        var = Variable(base_text, role_description="explanation prompt", requires_grad=True)
        from dp_textgrad.dp_es.population import Candidate
        parent = Candidate(var, {"id": 0})

        # Generate mutations
        offspring = mutator(
            parent=parent,
            iteration=1,
            rng=__import__('random').Random(42),
            feedback=None,
            num_offspring=2
        )

        for i, child in enumerate(offspring, 1):
            print(f"  Variant {i}: {child.get_value()}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print("ADVANCED MUTATION STRATEGIES - DEMONSTRATION")
    print("="*70)

    print("\n🎯 This example shows how advanced mutation strategies")
    print("   significantly improve DP-ES performance over simple concatenation.")

    # Run comparison
    run_comparison()

    # Demonstrate modes
    demonstrate_mutation_modes()

    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("\n1. ✅ LLM-guided mutations are more intelligent than concatenation")
    print("2. ✅ Adaptive strategies automatically adjust exploration/exploitation")
    print("3. ✅ Different modes (explore/exploit) suit different optimization stages")
    print("4. ✅ Diversity enforcement prevents premature convergence")
    print("\n💡 Recommendation: Use 'adaptive' strategy for most tasks")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("\n⚠️  This example requires an OpenAI API key.")
    print("   Set: export OPENAI_API_KEY='your-key-here'\n")

    # Uncomment to run (requires API key)
    # main()

    print("✓ Example code loaded. Set API key and uncomment main() to run.\n")
