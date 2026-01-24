OPENAI_API_KEY = ""
TOGETHERAI_API_KEY = ""
OPENROUTER_API_KEY = ""

OPENAI_MODEL_VERSION = "gpt-4o"
TOGETHERAI_MODEL_VERSION = ""
OPENROUTER_MODEL_VERSION = ""

EVIDENCE_RANKINGS_METHODS = ["iterative_similarity"] # Optional methods: "similarity","iterative_similarity","one_shot_prompting","iterative_prompting_sentence","nli_based_ranking","iterative_nli_based_ranking","one_shot_reason_rank_model","laquer_selection"

NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-large"
RERANKER_MODEL_NAME = "liuwenhan/reasonrank-7B"

CLAIM_TO_CANDIDATE_EVIDENCE_MAPPING_PATH = "data/claim_to_candidate_evidence_mapping.json"
INSTANCES_WITH_RANKED_EVIDENCE_PATH = ""

SHOULD_USE_GOLD_SNIPPETS = True

PLATFORM_TYPE = "openai"  # can be "togetherai", "openai", "open_router"

MAX_ATTEMPTS_FOR_API_QUERY = 3

MAX_ATTEMPTS_FOR_ITERATIVE_PROMPTING = 5

CONTRADICTION_INDEX = 0
ENTAILMENT_INDEX = 1
NEUTRAL_INDEX = 2

