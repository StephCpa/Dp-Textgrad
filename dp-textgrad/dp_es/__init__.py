"""
Differential Privacy aware Evolution Strategy utilities for TextGrad.

The dp_es package bundles together the following components:
    - Population management helpers (`population`)
    - Differentially private scoring utilities (`scorer`)
    - Privacy accounting (`accountant`)
    - Differentially private selection strategies (`selector`)
    - Mutation helpers that can leverage LLM based edits (`mutation`)
    - High level optimizer (`strategy`)

Users are expected to instantiate :class:`~dp_textgrad.dp_es.strategy.DPEvolutionStrategy`
with the components that fit their workflow.
"""

from .accountant import PrivacyAccountant, PrivacyBudgetExceeded, AdvancedCompositionAccountant
from .scorer import DPScorer, DPScorerConfig, DPScores
from .selector import DPSelectorConfig, DPSelector, DPSelectionResult, DPSelectionRecord
from .mutation import MutationEngine, MutationConfig
from .population import PopulationManager, Candidate
from .strategy import DPEvolutionStrategy, DPEvolutionConfig
from .critique import Critique, CritiqueOption, CritiquePipeline
from .feedback import FeedbackSanitiser, FeedbackSanitiserConfig
from .mutation_strategies import (
    MutationMode,
    LLMGuidedMutation,
    GradientGuidedMutation,
    AdaptiveMutation,
    CrossoverMutation,
    create_mutation_function,
)

__all__ = [
    "PrivacyAccountant",
    "PrivacyBudgetExceeded",
    "AdvancedCompositionAccountant",
    "DPScorer",
    "DPScorerConfig",
    "DPScores",
    "DPSelector",
    "DPSelectorConfig",
    "DPSelectionResult",
    "DPSelectionRecord",
    "MutationEngine",
    "MutationConfig",
    "PopulationManager",
    "Candidate",
    "DPEvolutionStrategy",
    "DPEvolutionConfig",
    "Critique",
    "CritiqueOption",
    "CritiquePipeline",
    "FeedbackSanitiser",
    "FeedbackSanitiserConfig",
    # Advanced mutation strategies
    "MutationMode",
    "LLMGuidedMutation",
    "GradientGuidedMutation",
    "AdaptiveMutation",
    "CrossoverMutation",
    "create_mutation_function",
]
