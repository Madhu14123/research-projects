"""
Configuration: paths, constants, EI dimension/feature definitions.
All weights and sub-feature assignments come from behavioral theory alone.
NO exam score involvement in Stages 1–4.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_ROOT   = r"D:\A_emotions in exam\data"
FACIAL_DIR  = os.path.join(DATA_ROOT, "facial_expressions", "facial expressions- data")
STAGE_DIR   = os.path.join(DATA_ROOT, "stage_coded_data",  "stage coded data")
SAM_FILE    = os.path.join(DATA_ROOT, "SAM Responses every 3 minutes.xlsx")
EXAM_FILE   = os.path.join(DATA_ROOT, "Exam Scores.xlsx")   # ONLY touched in Stage 5
OUTPUT_DIR  = r"D:\A_emotions in exam\outputs"

# ── AU columns (17 channels from OpenFace) ─────────────────────────────────
AU_COLS = [
    "AU01_r","AU02_r","AU04_r","AU05_r","AU06_r","AU07_r",
    "AU09_r","AU10_r","AU12_r","AU14_r","AU15_r","AU17_r",
    "AU20_r","AU23_r","AU25_r","AU26_r","AU45_r",
]

# ── Emotion composites (FACS-grounded AU summations) ───────────────────────
EMOTION_DEFS = {
    "Anger":     ["AU04_r","AU05_r","AU07_r","AU23_r"],
    "Fear":      ["AU01_r","AU04_r","AU05_r"],
    "Sadness":   ["AU01_r","AU04_r","AU15_r"],
    "Disgust":   ["AU04_r","AU07_r","AU09_r","AU10_r","AU17_r","AU20_r"],
    "Pride":     ["AU06_r","AU07_r","AU12_r"],
    "Happiness": ["AU06_r","AU12_r"],
    "Stress":    ["AU01_r","AU06_r","AU12_r","AU15_r"],
}
EMOTION_COLS   = list(EMOTION_DEFS.keys())
POSITIVE_EMOS  = ["Happiness", "Pride"]
NEGATIVE_EMOS  = ["Anger", "Fear", "Sadness", "Disgust", "Stress"]

# ── Stage label normalisation ──────────────────────────────────────────────
STAGE_NORM = {
    "block ": "Block", "block": "Block",
    "prod":   "Production", "production": "Production",
    "preparation": "Preparation",
    "ideation":    "Ideation",
    "evaluation":  "Evaluation",
    "refinement":  "Refinement",
}
PRODUCTIVE_STAGES = ["Production", "Ideation", "Evaluation", "Refinement", "Preparation"]
CORE_PRODUCTIVE   = ["Production", "Evaluation", "Refinement"]
BLOCK_STAGE       = "Block"
SAM_STAGE_PREFIX  = "sam"

# SAM checkpoint times (every 3 min for a 30-min session)
SAM_CHECKPOINT_TIMES_SEC = [i * 180 for i in range(1, 11)]  # 180,360,...,1800

# ── EI dimension weights (theory-grounded, from Process-Based Model PDF) ───
DIM_WEIGHTS = {
    "D1_EmotionPerception":      0.15,
    "D2_EmotionRegulation":      0.25,
    "D3_AdaptivePersistence":    0.25,
    "D4_AffectiveSelfAwareness": 0.15,
    "D5_EmotionCognitionAlign":  0.20,
}

# ── Candidate feature names (116 total, grouped by dimension) ──────────────
# Direction: +1 = higher raw value → higher EI;  -1 = higher raw value → lower EI (will be flipped)
# Used in composite construction ONLY — no exam scores involved.

CANDIDATE_FEATURES = {

    "D1_EmotionPerception": {
        # Emodiversity
        "c1_shannon_entropy":                  +1,
        "c1_simpson_diversity":                +1,
        "c1_gini_emodiversity":                -1,  # lower Gini = more equal = more diverse
        "c1_emotion_richness":                 +1,
        # Granularity via differentiation
        "c1_emotion_differentiation_icc":      +1,
        "c1_neg_emotion_differentiation":      +1,
        # Temporal dynamics
        "c1_emotion_transition_entropy":       +1,
        "c1_emotion_inertia_composite":        -1,  # high inertia = rigidity
        "c1_emotion_instability_mssd":         +1,  # moderate instability = responsiveness
        # Blending & co-occurrence
        "c1_mixed_emotion_rate":               +1,
        "c1_emotion_blend_count":              +1,
        # Cross-stage
        "c1_cross_stage_profile_diff":         +1,
        "c1_stage_emodiversity_variance":      +1,
        # Frequency domain
        "c1_spectral_entropy":                 +1,
        "c1_high_freq_power_ratio":            +1,
        # Distributional
        "c1_emotion_kurtosis":                 -1,  # lower kurtosis = richer
        "c1_skewness_balance":                 +1,
        # AU network
        "c1_au_activation_breadth":            +1,
        "c1_expression_distinctiveness":       +1,
        "c1_expression_ambiguity":             -1,  # lower = clearer
        "c1_au_network_density":               +1,
        "c1_posneg_balance":                   +1,  # from original set
    },

    "D2_EmotionRegulation": {
        # Recovery dynamics
        "c2_stress_recovery_latency":          -1,  # shorter = better
        "c2_recovery_completeness":            +1,
        "c2_peak_to_baseline_slope":           +1,
        # Emotional inertia
        "c2_stress_inertia":                   -1,
        "c2_negative_emotion_inertia":         -1,
        # Affective variability
        "c2_stress_instability_mssd":          -1,  # extreme instability = dysregulation
        "c2_emotion_variability_sd":           -1,
        # Regulation flexibility
        "c2_emotion_repertoire_diversity":     +1,
        "c2_cross_stage_regulation_variability": +1,
        "c2_regulation_strategy_switching":    +1,
        # Habituation
        "c2_stress_peak_attenuation":          +1,
        "c2_stress_habituation_ratio":         -1,  # lower ratio = better
        # AU suppression
        "c2_lip_press_index":                  +1,
        "c2_negative_expression_dampening":    +1,
        # Temporal regulation
        "c2_anticipatory_regulation":          +1,
        "c2_early_vs_late_regulation_ratio":   -1,  # lower = improving
        # Cross-stage
        "c2_block_stage_regulation_efficiency": +1,
        "c2_production_stress_floor":          -1,  # lower = better
        # Reappraisal
        "c2_neg_to_pos_transition_speed":      -1,  # faster = better
        "c2_reappraisal_proxy_coactivation":   +1,
        # Multi-channel
        "c2_sam_facial_discrepancy_reduction": +1,
        "c2_valence_arousal_regulation_coupling": +1,
    },

    "D3_AdaptivePersistence": {
        # Block-to-productive transitions
        "c3_block_exit_diversity":             +1,
        "c3_block_to_production_rate":         +1,
        "c3_recovery_acceleration":            +1,
        # Productive struggle
        "c3_productive_confusion_index":       +1,
        "c3_confusion_resolution_rate":        +1,
        # Advanced Markov
        "c3_productive_absorbing_probability": +1,
        "c3_second_order_block_escape":        +1,
        # Stage momentum
        "c3_forward_stage_momentum":           +1,
        # Engagement sustainability
        "c3_longest_productive_streak":        +1,
        "c3_productive_streak_variability":    -1,  # lower CV = more consistent
        "c3_late_task_engagement_ratio":       +1,
        # Frustration tolerance
        "c3_block_negative_affect_damping":    +1,
        "c3_block_anger_suppression":          -1,  # lower ratio = better
        "c3_stress_inertia_during_blocks":     -1,
        # Buoyancy
        "c3_bounce_back_index":                +1,
        "c3_adversity_composure_index":        +1,
        # Grit proxies
        "c3_total_productive_time_ratio":      +1,
        "c3_task_completion_persistence":      +1,
        "c3_effective_time_efficiency":        +1,
        # Stage quality
        "c3_productive_iteration_index":       +1,
        "c3_regressive_transition_ratio":      -1,
        "c3_block_emotional_trajectory_slope": +1,
        "c3_post_block_engagement_quality":    +1,
        "c3_struggle_benefit_index":           +1,
        # Original features
        "c3_markov_entropy":                   +1,
        "c3_stationary_block_inv":             +1,
    },

    "D4_AffectiveSelfAwareness": {
        # SAM-AU concordance (core)
        "c4_valence_happiness_concordance":    +1,
        "c4_valence_negemo_concordance":       +1,
        "c4_arousal_au_concordance":           +1,
        "c4_valence_stress_inverse_concordance": +1,
        # SAM trajectory
        "c4_valence_curvature":                +1,
        "c4_valence_turning_points":           +1,
        "c4_arousal_trend":                    -1,  # declining = optimizing
        "c4_arousal_stability":                +1,
        # Interoceptive accuracy
        "c4_facial_valence_accuracy":          +1,
        "c4_change_detection_alignment":       +1,
        "c4_block_stage_awareness":            +1,
        # Calibration
        "c4_positive_reporting_bias":          -1,
        "c4_absolute_calibration_error":       +1,  # already negated (higher = less error)
        # Arousal optimality
        "c4_arousal_optimality":               +1,
        "c4_arousal_midrange_proportion":      +1,
        # Self-efficacy
        "c4_valence_recovery_magnitude":       +1,
        "c4_pride_during_production":          +1,
        "c4_stage_emotion_appropriateness_d4": +1,
        "c4_evaluation_anxiety_alignment":     +1,
        # Alexithymia-reversed
        "c4_emotional_range_concordance":      +1,
        "c4_sam_response_range":               +1,
        "c4_anchoring_resistance":             +1,
        "c4_valence_arousal_discrimination":   +1,
        "c4_temporal_coherence":               +1,
        # Original features
        "c4_mean_valence":                     +1,
        "c4_valence_trend":                    +1,
    },

    "D5_EmotionCognitionAlign": {
        # Stage-appropriate emotion
        "c5_ideation_positive_affect":         +1,
        "c5_production_concentration":         +1,
        "c5_evaluation_critical_engagement":   +1,
        "c5_stage_emotion_profile_fit":        +1,
        # Emotion-stage coupling
        "c5_emotion_stage_differentiation":    +1,
        "c5_cognitive_effort_deployment":      +1,
        "c5_interest_during_ideation":         +1,
        # Flow state
        "c5_flow_proxy_production":            +1,
        "c5_flow_stability":                   +1,
        # Mastery mode
        "c5_mastery_mode_ratio":               +1,
        "c5_mastery_during_challenge":         +1,
        # Confusion-resolution
        "c5_confusion_resolution_rate_d5":     +1,
        "c5_block_constructive_confusion":     +1,
        # Stage transitions
        "c5_transition_emotional_preparation": +1,
        "c5_pride_after_production":           +1,
        # Task-appropriate arousal
        "c5_stress_appropriate_modulation":    +1,
        "c5_arousal_sam_task_match":           +1,
        # Cross-stage differentiation
        "c5_positive_creative_vs_focused_prod": +1,
        "c5_negative_affect_block_specificity": +1,
        "c5_emotion_complexity_index":         +1,
        "c5_sam_au_coherence":                 +1,
        "c5_engagement_recovery_speed":        +1,
    },
}

# Flat ordered list of all candidate feature names
ALL_CANDIDATES = [f for dim in CANDIDATE_FEATURES.values() for f in dim]

# Direction map: feature_name → +1 or -1 (used for EI-direction normalization)
FEATURE_DIRECTION = {f: d for dim in CANDIDATE_FEATURES.values() for f, d in dim.items()}
