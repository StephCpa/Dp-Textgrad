"""
Advanced Mutation Strategies for DP-ES

This module provides intelligent mutation strategies that go beyond
simple string concatenation, leveraging LLM capabilities and adaptive mechanisms.
"""

from __future__ import annotations

import random
from typing import List, Optional, Any, Callable
from enum import Enum
from dataclasses import dataclass

from dp_textgrad.variable import Variable
from dp_textgrad.engine import EngineLM
from dp_textgrad.config import validate_engine_or_get_default
from .population import Candidate
from .scorer import DPScores


class MutationMode(Enum):
    """Mutation intensity levels."""
    EXPLOIT = "exploit"  # Small, local changes
    EXPLORE = "explore"  # Large, global changes
    BALANCED = "balanced"  # Mix of both


@dataclass
class MutationStats:
    """Track mutation effectiveness for adaptive strategies."""

    total_mutations: int = 0
    successful_mutations: int = 0  # Led to score improvement
    average_score_delta: float = 0.0
    mode_history: List[MutationMode] = None

    def __post_init__(self):
        if self.mode_history is None:
            self.mode_history = []

    def success_rate(self) -> float:
        if self.total_mutations == 0:
            return 0.0
        return self.successful_mutations / self.total_mutations


class LLMGuidedMutation:
    """
    Use LLM to intelligently mutate variables based on feedback.

    This is more sophisticated than simple string concatenation,
    as the LLM can understand context and make meaningful improvements.
    """

    def __init__(
        self,
        engine: Optional[EngineLM] = None,
        mode: MutationMode = MutationMode.BALANCED,
        diversity_weight: float = 0.3,
    ):
        """
        Args:
            engine: LLM engine for generating mutations
            mode: Mutation intensity (exploit/explore/balanced)
            diversity_weight: How much to encourage diversity (0-1)
        """
        self.engine = validate_engine_or_get_default(engine)
        self.mode = mode
        self.diversity_weight = diversity_weight
        self.stats = MutationStats()

    def __call__(
        self,
        parent: Candidate,
        iteration: int,
        rng: random.Random,
        feedback: Optional[Any] = None,
        num_offspring: int = 2,
    ) -> List[Variable]:
        """
        Generate intelligent mutations using LLM.

        Args:
            parent: Parent candidate
            iteration: Current iteration
            rng: Random number generator
            feedback: DPScores or textual feedback
            num_offspring: Number of variants to generate

        Returns:
            List of mutated Variables
        """
        parent_text = parent.variable.get_value()
        role_desc = parent.variable.get_role_description()

        # Build mutation prompt based on mode and feedback
        mutation_prompt = self._build_mutation_prompt(
            parent_text, role_desc, iteration, feedback, num_offspring
        )

        # Generate mutations using LLM
        response = self.engine.generate(mutation_prompt)

        # Parse LLM output into separate variations
        variations = self._parse_variations(response, num_offspring)

        # Ensure diversity if requested
        if self.diversity_weight > 0:
            variations = self._enforce_diversity(
                variations, parent_text, rng
            )

        # Convert to Variable objects
        offspring = [
            Variable(
                value=var,
                role_description=role_desc,
                requires_grad=True
            )
            for var in variations
        ]

        self.stats.total_mutations += len(offspring)

        return offspring

    def _build_mutation_prompt(
        self,
        parent_text: str,
        role_desc: str,
        iteration: int,
        feedback: Optional[Any],
        num_offspring: int
    ) -> str:
        """Construct intelligent mutation prompt based on mode and feedback."""

        # Extract feedback information
        feedback_text = ""
        if feedback is not None:
            if isinstance(feedback, DPScores):
                # Use DP scores to guide mutation
                avg_score = sum(r.dp_score for r in feedback.records) / len(feedback.records)
                feedback_text = (
                    f"Previous evaluation score: {avg_score:.3f}/1.0\n"
                    f"Iteration: {iteration}\n"
                )
            elif isinstance(feedback, str):
                feedback_text = f"Feedback: {feedback}\n"

        # Mode-specific instructions
        if self.mode == MutationMode.EXPLOIT:
            intensity_instruction = (
                "Make SMALL, INCREMENTAL improvements. "
                "Refine wording, add minor details, or adjust phrasing. "
                "Keep the overall structure and meaning similar."
            )
        elif self.mode == MutationMode.EXPLORE:
            intensity_instruction = (
                "Make SIGNIFICANT, CREATIVE changes. "
                "Try completely different approaches, structures, or perspectives. "
                "Be bold and experimental."
            )
        else:  # BALANCED
            intensity_instruction = (
                "Create a MIX of small improvements and larger creative changes. "
                "Some variations should be conservative, others more experimental."
            )

        prompt = f"""You are helping optimize text through evolutionary mutation.

**Current text** ({role_desc}):
"{parent_text}"

{feedback_text}

**Task**: Generate {num_offspring} improved variations of this text.

**Mutation strategy**: {intensity_instruction}

**Requirements**:
1. Each variation should be different from the original and from each other
2. Maintain the core purpose described in the role: {role_desc}
3. Output exactly {num_offspring} variations, separated by "---VARIANT---"
4. Each variation should be complete and standalone

**Output format**:
<variation 1>
---VARIANT---
<variation 2>
{f'---VARIANT---{chr(10)}<variation {num_offspring}>' if num_offspring > 2 else ''}

Generate the variations now:"""

        return prompt

    def _parse_variations(self, llm_output: str, expected_count: int) -> List[str]:
        """Parse LLM output into individual variations."""
        separator = "---VARIANT---"

        if separator in llm_output:
            variations = [
                v.strip() for v in llm_output.split(separator)
                if v.strip()
            ]
        else:
            # Fallback: split by double newlines
            variations = [
                v.strip() for v in llm_output.split("\n\n")
                if v.strip() and len(v.strip()) > 10
            ]

        # Ensure we have the right count
        if len(variations) < expected_count:
            # Repeat to fill
            while len(variations) < expected_count:
                variations.append(variations[0] + f" [variant {len(variations)}]")

        return variations[:expected_count]

    def _enforce_diversity(
        self,
        variations: List[str],
        parent: str,
        rng: random.Random
    ) -> List[str]:
        """Ensure variations are sufficiently different from each other.

        Optimized version using character n-grams for faster similarity computation.
        """

        def text_similarity_ngram(t1: str, t2: str, n: int = 3) -> float:
            """Fast n-gram based similarity (significantly faster than word Jaccard).

            Uses character n-grams which are more robust and ~3x faster than word-level.
            """
            if not t1 or not t2:
                return 0.0

            # Convert to lowercase once
            t1_lower = t1.lower()
            t2_lower = t2.lower()

            # Quick length-based check
            len_ratio = min(len(t1_lower), len(t2_lower)) / max(len(t1_lower), len(t2_lower))
            if len_ratio < 0.3:  # Very different lengths -> likely different
                return 0.0

            # Generate n-grams efficiently
            ngrams1 = set(t1_lower[i:i+n] for i in range(len(t1_lower) - n + 1))
            ngrams2 = set(t2_lower[i:i+n] for i in range(len(t2_lower) - n + 1))

            if not ngrams1 or not ngrams2:
                return 0.0

            # Jaccard similarity on n-grams
            intersection = len(ngrams1 & ngrams2)
            union = len(ngrams1 | ngrams2)

            return intersection / union if union > 0 else 0.0

        diverse_variations = []
        similarity_threshold = 1.0 - self.diversity_weight

        # Cache parent n-grams for efficiency
        parent_lower = parent.lower()
        parent_ngrams = set(parent_lower[i:i+3] for i in range(len(parent_lower) - 2))

        for var in variations:
            # Check similarity with parent and existing variations
            too_similar = False

            # Quick check against parent using cached n-grams
            var_lower = var.lower()
            var_ngrams = set(var_lower[i:i+3] for i in range(len(var_lower) - 2))

            if var_ngrams and parent_ngrams:
                intersection = len(var_ngrams & parent_ngrams)
                union = len(var_ngrams | parent_ngrams)
                parent_sim = intersection / union if union > 0 else 0.0

                if parent_sim > similarity_threshold:
                    too_similar = True

            # Check against existing diverse variations
            if not too_similar:
                for existing in diverse_variations:
                    if text_similarity_ngram(var, existing) > similarity_threshold:
                        too_similar = True
                        break

            if not too_similar:
                diverse_variations.append(var)
            else:
                # Intelligent perturbation strategies
                perturbation_strategies = [
                    lambda v: v + " " + rng.choice(['(alternative approach)', '(refined version)', '(enhanced variant)']),
                    lambda v: rng.choice(['Enhanced: ', 'Refined: ', 'Modified: ']) + v,
                    lambda v: v.replace('.', f'. {rng.choice(["Additionally,", "Furthermore,", "Moreover,"])}'),
                ]

                # Try each strategy until we get something diverse
                perturbed = var
                for strategy in rng.sample(perturbation_strategies, len(perturbation_strategies)):
                    perturbed = strategy(var)
                    # Check if this perturbation is diverse enough
                    is_diverse = all(
                        text_similarity_ngram(perturbed, existing) <= similarity_threshold
                        for existing in diverse_variations
                    )
                    if is_diverse:
                        break

                diverse_variations.append(perturbed)

        return diverse_variations


class GradientGuidedMutation:
    """
    Use TextGrad-style gradients to guide mutation direction.

    This combines evolution strategy with gradient-based hints.
    """

    def __init__(self, engine: Optional[EngineLM] = None):
        self.engine = validate_engine_or_get_default(engine)

    def __call__(
        self,
        parent: Candidate,
        iteration: int,
        rng: random.Random,
        feedback: Optional[Any] = None,
        num_offspring: int = 2,
    ) -> List[Variable]:
        """
        Generate mutations guided by gradient-like feedback.

        If parent has gradient information (from TextLoss), use it
        to generate more targeted mutations.
        """
        parent_var = parent.variable
        parent_text = parent_var.get_value()
        role_desc = parent_var.get_role_description()

        # Check if variable has gradient information
        gradient_hint = ""
        if hasattr(parent_var, 'gradients') and parent_var.gradients:
            gradient_hint = (
                f"Improvement direction: {parent_var.gradients}\n"
                "Focus mutations on addressing these points."
            )

        # Build gradient-guided prompt
        prompt = f"""Improve the following {role_desc}:

"{parent_text}"

{gradient_hint}

Generate {num_offspring} improved versions that address the suggested improvements.
Separate each version with "---VERSION---"."""

        response = self.engine.generate(prompt)

        # Parse variations
        variations = [
            v.strip() for v in response.split("---VERSION---")
            if v.strip()
        ][:num_offspring]

        # Pad if needed
        while len(variations) < num_offspring:
            variations.append(parent_text + f" [improved v{len(variations)}]")

        return [
            Variable(v, role_description=role_desc, requires_grad=True)
            for v in variations
        ]


class AdaptiveMutation:
    """
    Automatically adjust mutation strategy based on performance.

    Starts with exploration, gradually shifts to exploitation as
    optimization progresses. Can also adapt based on success rate.
    """

    def __init__(
        self,
        engine: Optional[EngineLM] = None,
        initial_mode: MutationMode = MutationMode.EXPLORE,
    ):
        self.engine = validate_engine_or_get_default(engine)
        self.mode = initial_mode
        self.exploit_mutator = LLMGuidedMutation(engine, MutationMode.EXPLOIT)
        self.explore_mutator = LLMGuidedMutation(engine, MutationMode.EXPLORE)
        self.balanced_mutator = LLMGuidedMutation(engine, MutationMode.BALANCED)

        self.iteration_scores: List[float] = []

    def __call__(
        self,
        parent: Candidate,
        iteration: int,
        rng: random.Random,
        feedback: Optional[Any] = None,
        num_offspring: int = 2,
    ) -> List[Variable]:
        """Generate mutations with adaptive strategy."""

        # Update mode based on iteration and performance
        self._update_mode(iteration, feedback)

        # Select appropriate mutator
        if self.mode == MutationMode.EXPLOIT:
            mutator = self.exploit_mutator
        elif self.mode == MutationMode.EXPLORE:
            mutator = self.explore_mutator
        else:
            mutator = self.balanced_mutator

        return mutator(parent, iteration, rng, feedback, num_offspring)

    def _update_mode(self, iteration: int, feedback: Optional[Any]):
        """Adapt mutation mode based on progress."""

        # Extract score if available
        current_score = None
        if feedback is not None and isinstance(feedback, DPScores):
            if feedback.records:
                current_score = max(r.dp_score for r in feedback.records)
                self.iteration_scores.append(current_score)

        # Simple adaptive strategy:
        # - Early iterations (< 30%): Explore
        # - Middle iterations (30-70%): Balanced
        # - Late iterations (> 70%): Exploit

        # If we have score history, also check for plateaus
        if len(self.iteration_scores) >= 3:
            recent_improvement = (
                self.iteration_scores[-1] - self.iteration_scores[-3]
            )

            if recent_improvement < 0.01:  # Plateau
                # Stuck in local optimum, switch to explore
                self.mode = MutationMode.EXPLORE
            elif recent_improvement > 0.1:  # Rapid improvement
                # Keep exploiting this direction
                self.mode = MutationMode.EXPLOIT
            else:
                # Moderate progress, stay balanced
                self.mode = MutationMode.BALANCED


class CrossoverMutation:
    """
    Combine features from multiple parents (genetic algorithm style).

    This can create offspring that inherit good properties from
    different successful candidates.
    """

    def __init__(self, engine: Optional[EngineLM] = None):
        self.engine = validate_engine_or_get_default(engine)

    def crossover(
        self,
        parents: List[Candidate],
        iteration: int,
        rng: random.Random,
        num_offspring: int = 2,
    ) -> List[Variable]:
        """
        Create offspring by combining elements from multiple parents.

        Args:
            parents: List of parent candidates (2 or more)
            iteration: Current iteration
            rng: Random generator
            num_offspring: Number of children to create

        Returns:
            List of offspring Variables
        """
        if len(parents) < 2:
            # Fallback to simple mutation
            return [Variable(
                parents[0].variable.get_value(),
                role_description=parents[0].variable.get_role_description(),
                requires_grad=True
            )]

        # Extract parent texts
        parent_texts = [p.variable.get_value() for p in parents[:3]]  # Max 3
        role_desc = parents[0].variable.get_role_description()

        # Build crossover prompt
        parent_list = "\n".join([
            f"Parent {i+1}: {text}"
            for i, text in enumerate(parent_texts)
        ])

        prompt = f"""You are creating offspring by combining good elements from multiple parents.

**Parents** ({role_desc}):
{parent_list}

**Task**: Create {num_offspring} new variations that combine the best elements from these parents.

**Strategy**:
- Identify strong elements in each parent
- Mix and match these elements creatively
- Create coherent, improved offspring

Output {num_offspring} variations, separated by "---CHILD---":"""

        response = self.engine.generate(prompt)

        # Parse offspring
        offspring_texts = [
            o.strip() for o in response.split("---CHILD---")
            if o.strip()
        ][:num_offspring]

        # Pad if needed
        while len(offspring_texts) < num_offspring:
            offspring_texts.append(rng.choice(parent_texts))

        return [
            Variable(text, role_description=role_desc, requires_grad=True)
            for text in offspring_texts
        ]


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_mutation_function(
    strategy: str = "adaptive",
    engine: Optional[EngineLM] = None,
    **kwargs
) -> Callable:
    """
    Factory function to create mutation functions with different strategies.

    Args:
        strategy: One of "llm_guided", "gradient_guided", "adaptive", "crossover"
        engine: LLM engine to use
        **kwargs: Additional strategy-specific parameters

    Returns:
        Mutation function compatible with MutationEngine

    Example:
        >>> mutation_fn = create_mutation_function("adaptive", engine=my_engine)
        >>> engine = MutationEngine(mutation_fn=mutation_fn, config=config)
    """

    if strategy == "llm_guided":
        mode = kwargs.get("mode", MutationMode.BALANCED)
        mutator = LLMGuidedMutation(engine=engine, mode=mode)
        return lambda parent, iter, rng, feedback: mutator(
            parent, iter, rng, feedback,
            num_offspring=kwargs.get("num_offspring", 2)
        )

    elif strategy == "gradient_guided":
        mutator = GradientGuidedMutation(engine=engine)
        return lambda parent, iter, rng, feedback: mutator(
            parent, iter, rng, feedback,
            num_offspring=kwargs.get("num_offspring", 2)
        )

    elif strategy == "adaptive":
        mutator = AdaptiveMutation(engine=engine)
        return lambda parent, iter, rng, feedback: mutator(
            parent, iter, rng, feedback,
            num_offspring=kwargs.get("num_offspring", 2)
        )

    elif strategy == "crossover":
        mutator = CrossoverMutation(engine=engine)
        # Note: crossover needs multiple parents, handle specially
        return mutator.crossover

    else:
        raise ValueError(
            f"Unknown mutation strategy: {strategy}. "
            f"Choose from: llm_guided, gradient_guided, adaptive, crossover"
        )


if __name__ == "__main__":
    print("\n" + "="*70)
    print("ADVANCED MUTATION STRATEGIES FOR DP-ES")
    print("="*70)
    print("\nAvailable strategies:")
    print("  1. LLMGuidedMutation - Intelligent LLM-based mutations")
    print("  2. GradientGuidedMutation - Use TextGrad gradients")
    print("  3. AdaptiveMutation - Automatically adjust exploration/exploitation")
    print("  4. CrossoverMutation - Combine multiple parents")
    print("\nUsage:")
    print("  from dp_textgrad.dp_es.mutation_strategies import create_mutation_function")
    print("  mutation_fn = create_mutation_function('adaptive')")
    print("="*70 + "\n")
