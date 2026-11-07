"""ECA runtime for queue/analyzer supervision.

Extended in v0.2 with the Goal-Conditioned Structural Time Model:
  - StructuralTimeTracker: stateful tracker for z_t, tau, ECA, PSA
  - GoalConditionedState: full latent structural state (m_t, g_t, p_t, u_t)
  - build_goal_conditioned_state: convenience integration function

Extended with D_flow (Tier 0/1/2 event-driven structural flow estimator):
  - run_d_flow_pipeline: main entry point for D_flow Tier 0/1/2
  - compute_residual_summary: Tier 0 cheap residual summary (every turn)
  - compute_d_flow_tier1: Tier 1 sparse unbalanced Sinkhorn (conditional)
  - compute_d_flow_tier2: Tier 2 local refinement (decision boundary only)
  - compute_q_flow: D_flow confidence estimator
  - decide_tier: Tier 0/1/2 transition logic
  - compute_D_O_with_flow: D_core + D_flow blending via q_flow
  - DFlowResult, ResidualSummary, SinkhornResult, TierDecision
"""

from .runtime import ECARuntime
from .types import AnxietySnapshot, DriveWeights, ECAConfig, ECAState, InterruptDecision, NightBundle
from .structural_model import (
    GoalConditionedState,
    GoalHypothesis,
    StructuralModel,
    ObjectFeature,
    RelationEdge,
    InvariantFeature,
    AffordanceHeatmap,
    compute_structural_diff,
    compute_structural_eca,
    compute_structural_psa,
    compute_action_utility,
    build_structural_model_from_grid,
    generate_goal_hypothesis,
    update_internal_time,
    compute_adaptive_epsilon,
    phi_meaningful,
)
from .observer import (
    StructuralTimeTracker,
    build_goal_conditioned_state,
)
from .d_flow import (
    DFlowResult,
    ResidualSummary,
    SinkhornResult,
    TierDecision,
    compute_residual_summary,
    compute_d_flow_tier1,
    compute_d_flow_tier2,
    compute_q_flow,
    decide_tier,
    compute_D_O_with_flow,
    compute_D_I_stabilized,
    compute_affordance_entropy_delta,
    run_d_flow_pipeline,
)
from .action_effect_model import (
    ActionEffectRecord,
    ActionEffectStats,
    ActionEffectSummary,
    ActionEffectTracker,
    ChangedBBox,
    PrimitiveClassification,
    PrimitiveClassifier,
    frame_state_hash,
)
from .action_effect_hypotheses import (
    ActionEffectHypothesis,
    ActionEffectHypothesisBuilder,
)
from .perceptual_state_abstraction import (
    PSAAbstraction,
    PerceptualStateAbstractionInducer,
)
from .internal_llm_cognition import (
    ECA_INTERPRETER_SCHEMA_VERSION,
    InternalLLMCognitionBridge,
    InternalLLMCognitionResult,
    InternalLLMInterpretationResult,
    InternalLLMInterpreterBridge,
    InternalLLMPromptPacket,
    PSA_PROVIDER_SCHEMA_VERSION,
    build_eca_llm_interpreter_prompt_packet,
    build_psa_llm_prompt_packet,
    internal_llm_prompt_packet_audit,
    parse_internal_llm_json_response,
    resolve_internal_llm_provider_path,
)
from .internal_llm_ollama_provider import (
    DEFAULT_GEMMA_E4B_MODEL,
    ECA_GEMMA_E4B_CALL_SITE,
    PSA_GEMMA_E4B_CALL_SITE,
    eca_gemma_e4b_provider,
    psa_gemma_e4b_provider,
)
from .primitive_selector import (
    PrimitiveCandidate,
    PrimitiveSelection,
    SolverPrimitiveSelector,
)
from .primitive_synthesis import (
    CORE_ACTION_ORDER,
    DIRECTIONAL_ACTIONS,
    PrimitiveActionDecision,
    PrimitiveSynthesisLoop,
)
from .structural_rods import (
    CausalTraceRod,
    EffectEvent,
    HypothesisRod,
    HypothesisRodMemory,
    RodPromotionPolicy,
    SpatialRodPosterior,
    SpatialRodPosteriorMemory,
    StructuralRodMemory,
    build_causal_trace_rod,
    build_effect_event,
    build_hypothesis_rods_from_causal_pass_result,
    default_rod_promotion_policy,
    infer_effect_type,
)
from .policy_bundle import (
    build_runtime_policy_packet,
    collect_policy_examples,
    evaluate_policy_bundle,
    load_policy_bundle,
    predict_action,
    predict_candidate_score,
    predict_mode,
    predict_risk,
    train_policy_bundle,
    train_policy_bundle_from_roots,
    write_policy_bundle,
)
from .generalisation_proxy import build_generalisation_proxy_report
from .adapter_mismatch_learner import (
    build_repair_candidate_ranker_report,
    build_repair_candidate_ranker_report_from_roots,
    build_repair_distillation_report,
    build_repair_distillation_report_from_roots,
    build_repair_student_shadow_report,
    build_repair_student_shadow_report_from_roots,
    evaluate_repair_candidate_ranker,
    extract_repair_distillation_examples,
    evaluate_repair_payload_student,
    write_repair_candidate_ranker_report,
    write_repair_distillation_report,
    write_repair_student_shadow_report,
)
from .perturbation_benchmark import (
    PerturbationSpec,
    build_perturbation_specs,
    run_perception_perturbation_benchmark,
    run_perception_perturbation_episode,
    summarise_perturbation_rows,
)

__all__ = [
    # Original ECA exports
    "AnxietySnapshot",
    "DriveWeights",
    "ECAConfig",
    "ECAState",
    "ECARuntime",
    "InterruptDecision",
    "NightBundle",
    # v0.2: Goal-Conditioned Structural Time Model
    "GoalConditionedState",
    "GoalHypothesis",
    "StructuralModel",
    "ObjectFeature",
    "RelationEdge",
    "InvariantFeature",
    "AffordanceHeatmap",
    "StructuralTimeTracker",
    "build_goal_conditioned_state",
    "build_structural_model_from_grid",
    "compute_structural_diff",
    "compute_structural_eca",
    "compute_structural_psa",
    "compute_action_utility",
    "generate_goal_hypothesis",
    "update_internal_time",
    "compute_adaptive_epsilon",
    "phi_meaningful",
    # D_flow: Tier 0/1/2 event-driven structural flow estimator
    "DFlowResult",
    "ResidualSummary",
    "SinkhornResult",
    "TierDecision",
    "compute_residual_summary",
    "compute_d_flow_tier1",
    "compute_d_flow_tier2",
    "compute_q_flow",
    "decide_tier",
    "compute_D_O_with_flow",
    "compute_D_I_stabilized",
    "compute_affordance_entropy_delta",
    "run_d_flow_pipeline",
    # Phase 23.X: frame-only action-effect memory and primitive classifier
    "ActionEffectRecord",
    "ActionEffectStats",
    "ActionEffectSummary",
    "ActionEffectTracker",
    "ChangedBBox",
    "PrimitiveClassification",
    "PrimitiveClassifier",
    "frame_state_hash",
    "ActionEffectHypothesis",
    "ActionEffectHypothesisBuilder",
    "PSAAbstraction",
    "PerceptualStateAbstractionInducer",
    "ECA_INTERPRETER_SCHEMA_VERSION",
    "InternalLLMCognitionBridge",
    "InternalLLMCognitionResult",
    "InternalLLMInterpretationResult",
    "InternalLLMInterpreterBridge",
    "InternalLLMPromptPacket",
    "PSA_PROVIDER_SCHEMA_VERSION",
    "build_eca_llm_interpreter_prompt_packet",
    "build_psa_llm_prompt_packet",
    "internal_llm_prompt_packet_audit",
    "parse_internal_llm_json_response",
    "resolve_internal_llm_provider_path",
    "DEFAULT_GEMMA_E4B_MODEL",
    "ECA_GEMMA_E4B_CALL_SITE",
    "PSA_GEMMA_E4B_CALL_SITE",
    "eca_gemma_e4b_provider",
    "psa_gemma_e4b_provider",
    "PrimitiveCandidate",
    "PrimitiveSelection",
    "SolverPrimitiveSelector",
    "CORE_ACTION_ORDER",
    "DIRECTIONAL_ACTIONS",
    "PrimitiveActionDecision",
    "PrimitiveSynthesisLoop",
    "CausalTraceRod",
    "EffectEvent",
    "HypothesisRod",
    "HypothesisRodMemory",
    "RodPromotionPolicy",
    "SpatialRodPosterior",
    "SpatialRodPosteriorMemory",
    "StructuralRodMemory",
    "build_causal_trace_rod",
    "build_effect_event",
    "build_hypothesis_rods_from_causal_pass_result",
    "default_rod_promotion_policy",
    "infer_effect_type",
    "build_runtime_policy_packet",
    "collect_policy_examples",
    "evaluate_policy_bundle",
    "load_policy_bundle",
    "predict_action",
    "predict_candidate_score",
    "predict_mode",
    "predict_risk",
    "train_policy_bundle",
    "train_policy_bundle_from_roots",
    "write_policy_bundle",
    "build_generalisation_proxy_report",
    "build_repair_candidate_ranker_report",
    "build_repair_candidate_ranker_report_from_roots",
    "build_repair_distillation_report",
    "build_repair_distillation_report_from_roots",
    "build_repair_student_shadow_report",
    "build_repair_student_shadow_report_from_roots",
    "evaluate_repair_candidate_ranker",
    "evaluate_repair_payload_student",
    "extract_repair_distillation_examples",
    "write_repair_candidate_ranker_report",
    "write_repair_distillation_report",
    "write_repair_student_shadow_report",
    "PerturbationSpec",
    "build_perturbation_specs",
    "run_perception_perturbation_benchmark",
    "run_perception_perturbation_episode",
    "summarise_perturbation_rows",
]
