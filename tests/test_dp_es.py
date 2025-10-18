import random

import pytest

from dp_textgrad import Variable
from dp_textgrad.dp_es import (
    DPEvolutionStrategy,
    DPEvolutionConfig,
    PrivacyAccountant,
    AdvancedCompositionAccountant,
    DPScorer,
    DPScorerConfig,
    DPSelector,
    DPSelectorConfig,
    MutationEngine,
    MutationConfig,
    Critique,
    CritiquePipeline,
)
from dp_textgrad.dp_es.population import Candidate


def test_dp_scorer_clipping_and_noise():
    variable = Variable("value", role_description="test variable", requires_grad=False)
    candidate = Candidate(variable=variable, metadata={"candidate_id": "c0"})
    config = DPScorerConfig(clipping_value=1.0, noise_multiplier=0.0, epsilon=0.5, delta=1e-5)
    scorer = DPScorer(config)

    def evaluation_fn(cand: Candidate) -> float:
        return 10.0  # deliberately above clipping threshold

    scores = scorer.evaluate([candidate], evaluation_fn, rng=random.Random(0))
    assert scores.records[0].raw_score == pytest.approx(10.0)
    assert scores.records[0].clipped_score == pytest.approx(1.0)
    assert scores.records[0].dp_score == pytest.approx(1.0)  # no noise
    assert scores.epsilon == pytest.approx(0.5)
    assert scores.delta == pytest.approx(1e-5)
    updated_candidate = scores.updated_candidates[0]
    assert updated_candidate.metadata["dp_last_clipped_score"] == pytest.approx(1.0)


def test_dp_scorer_feedback_and_auto_noise():
    variable = Variable("value", role_description="test variable", requires_grad=False)
    candidate = Candidate(variable=variable, metadata={"candidate_id": "c1"})
    config = DPScorerConfig(clipping_value=2.0, noise_multiplier=None, epsilon=0.8, delta=1e-4)
    scorer = DPScorer(config)

    feedback_text = "looks good"

    def evaluation_fn(cand: Candidate):
        return 1.5, feedback_text

    scores = scorer.evaluate([candidate], evaluation_fn, rng=random.Random(0), description="round-0")
    assert scores.description == "round-0"
    assert scores.records[0].feedback == feedback_text
    updated = scores.updated_candidates[0]
    assert updated.metadata["dp_feedback"] == feedback_text
    assert config.noise_multiplier > 0  # auto calibrated


def test_dp_evolution_strategy_improves_candidates():
    target = Variable("0", role_description="optimisation target", requires_grad=True)

    scorer = DPScorer(DPScorerConfig(clipping_value=10.0, noise_multiplier=0.0, epsilon=0.5, delta=1e-5))
    selector = DPSelector(DPSelectorConfig(select_k=2, epsilon=0.0, gumbel_scale_override=0.0))

    mutation_config = MutationConfig(offspring_per_parent=2, allow_identity_offspring=True)
    captured_feedback = []

    def mutation_fn(parent: Candidate, iteration: int, rng: random.Random, feedback):
        captured_feedback.append(feedback)
        base = int(parent.variable.get_value())
        children = []
        for delta in range(1, mutation_config.offspring_per_parent + 1):
            child_value = str(base + delta)
            child = Variable(child_value, role_description=parent.variable.get_role_description(), requires_grad=True)
            children.append(child)
        return children

    mutation_engine = MutationEngine(mutation_fn=mutation_fn, config=mutation_config)

    accountant = PrivacyAccountant(target_epsilon=5.0, target_delta=1e-4)

    def evaluation_fn(var: Variable) -> float:
        return float(int(var.get_value()))

    strategy = DPEvolutionStrategy(
        parameter=target,
        evaluation_fn=evaluation_fn,
        scorer=scorer,
        selector=selector,
        mutation_engine=mutation_engine,
        accountant=accountant,
        config=DPEvolutionConfig(
            population_size=3,
            parents_to_select=2,
            max_iterations=3,
            rng_seed=123,
        ),
    )

    strategy.step()

    assert int(target.get_value()) >= 3
    assert accountant.consumed_epsilon == pytest.approx(1.5)  # 3 iterations * 0.5 epsilon from scoring
    assert captured_feedback  # ensure mutation received DPScores
    assert all(hasattr(item, "records") for item in captured_feedback if item is not None)


def test_dp_selector_reports_privacy_cost_and_noise():
    variable = Variable("value", role_description="candidate", requires_grad=False)
    base_candidate = Candidate(variable=variable, metadata={"candidate_id": "cand-0"})
    scored_candidate = base_candidate.with_scores(raw_score=1.0, dp_score=1.0, noise=0.0)
    other_candidate = base_candidate.with_scores(raw_score=0.5, dp_score=0.5, noise=0.0)

    config = DPSelectorConfig(select_k=1, epsilon=0.5, sensitivity=1.0)
    selector = DPSelector(config)

    result = selector.select_with_metadata([scored_candidate, other_candidate], rng=random.Random(42))

    assert result.selected  # at least one selected
    assert result.epsilon == pytest.approx(0.5)
    assert result.delta == pytest.approx(0.0)
    # 由于加入了 Gumbel 噪声，记录中应存在非零噪声（概率事件；使用固定种子确保结果稳定）
    assert any(record.noise != 0.0 for record in result.records)


def test_mutation_engine_records_selected_critique():
    variable = Variable("seed", role_description="candidate", requires_grad=True)
    parent = Candidate(variable=variable, metadata={"candidate_id": "cand-root"})

    def critique_generator(parent_candidate, iteration, rng, feedback):
        return [
            Critique(text="Consider expanding with detail.", metadata={"rank": 0}),
            Critique(text="Great job!", metadata={"rank": 1}),
        ]

    def critique_evaluation(parent_candidate, critique: Critique):
        return 1.0 if "Great" in critique.text else 0.2

    critique_scorer = DPScorer(DPScorerConfig(clipping_value=1.0, noise_multiplier=0.0, epsilon=0.3, delta=1e-6))
    critique_selector = DPSelector(DPSelectorConfig(select_k=1, epsilon=0.0, sensitivity=1.0))
    pipeline = CritiquePipeline(
        generator_fn=critique_generator,
        evaluation_fn=critique_evaluation,
        dp_scorer=critique_scorer,
        dp_selector=critique_selector,
    )

    mutation_config = MutationConfig(offspring_per_parent=1, allow_identity_offspring=True)

    def mutation_fn(parent_candidate, iteration, rng, feedback):
        critique_text = parent_candidate.metadata.get("dp_selected_critique", "")
        child_value = parent_candidate.variable.get_value() + "|" + critique_text
        return [
            Variable(
                child_value,
                role_description=parent_candidate.variable.get_role_description(),
                requires_grad=True,
            )
        ]

    engine = MutationEngine(mutation_fn=mutation_fn, config=mutation_config, critique_pipeline=pipeline)
    accountant = PrivacyAccountant(target_epsilon=1.0, target_delta=1e-5)
    engine.bind_accountant(accountant)

    offspring = engine.generate(
        parents=[parent],
        iteration=1,
        rng=random.Random(42),
        feedback=None,
    )

    assert parent.metadata["dp_selected_critique"] == "Great job!"
    assert len(offspring) == 1
    assert "Great job!" in offspring[0].variable.get_value()
    assert pytest.approx(accountant.consumed_epsilon, rel=1e-6) == 0.3


def test_advanced_accountant_allows_smaller_epsilon_budget():
    accountant = AdvancedCompositionAccountant(target_epsilon=1.5, target_delta=1e-4, delta_slack=1e-6)
    accountant.consume(0.4, 1e-5, description="step-1")
    accountant.consume(0.4, 1e-5, description="step-2")
    remaining_epsilon, remaining_delta = accountant.remaining_budget()
    assert remaining_epsilon < 1.5
    assert remaining_delta < 1e-4
