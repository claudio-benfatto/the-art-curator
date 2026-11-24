variable "aws_region" {
  description = "AWS region for Bedrock and other resources. Must match config.py's AWS_REGION default (eu-west-1) unless overridden there too."
  type        = string
  default     = "eu-west-1"
}

variable "github_repo" {
  description = "GitHub repo (org/name) allowed to assume the CI role via OIDC."
  type        = string
  default     = "claudio-benfatto/the-art-curator"
}

# GitHub issues OIDC tokens with immutable subjects for this repo:
# `repo:<owner>@<owner_id>/<repo>@<repo_id>:<context>`. The numeric IDs mean a
# renamed or deleted-and-recreated repo can't satisfy this trust policy.
# Read the current value with:
#   gh api repos/<owner>/<repo>/actions/oidc/customization/sub --jq .sub_claim_prefix
variable "github_oidc_sub_prefix" {
  description = "Prefix of the GitHub OIDC `sub` claim for this repo (immutable-subject format). Must match GitHub exactly, or every CI role assumption is denied."
  type        = string
  default     = "repo:claudio-benfatto@7601067/the-art-curator@1376843713"
}

variable "apply_environment" {
  description = "GitHub environment whose OIDC subject may assume the apply role. Must match the `environment:` of the apply job in .github/workflows/terraform.yml, and must have a required reviewer (infra/README.md step 4)."
  type        = string
  default     = "infra-apply"
}

variable "smoke_environment" {
  description = "GitHub environment whose OIDC subject may assume the smoke role. Must match the `environment:` of .github/workflows/smoke.yml (infra/README.md step 5)."
  type        = string
  default     = "smoke"
}

variable "tf_state_bucket" {
  description = "Name of the hand-created S3 state bucket (infra/README.md step 1). No default — must match backend.tf's -backend-config bucket exactly, so the CI role's permissions target the right ARN."
  type        = string
}

# Keep these in sync with config.py's CHAT_MODEL / EXTRACT_MODEL defaults.
# EMBED_MODEL is chosen by measurement in P3 from embed_candidate_model_ids.

# Both are bedrock-runtime *catalog* IDs, called through their `global.`
# inference profiles, so config.py's CHAT_MODEL / EXTRACT_MODEL are these with
# a `global.` prefix. Mantle is blocked for this account (CLAUDE.md, Current
# state); chat sits on Opus 4.6, the newest Claude this account can call.
variable "chat_model_id" {
  description = "Bedrock catalog model ID for chat (Claude only, per CLAUDE.md #4), invoked via its global. inference profile."
  type        = string
  default     = "anthropic.claude-opus-4-6-v1"
}

# quotas.tf. The account default is 5, which one busy conversation exceeds.
# Keep config.py's REQUESTS_PER_MINUTE at what AWS actually grants.
variable "chat_requests_per_minute" {
  description = "Requests a minute to ask AWS for on chat_model_id's global cross-region quota. Above the current limit, an apply files an increase request."
  type        = number
  default     = 50
}

variable "extract_model_id" {
  description = "Bedrock catalog model ID for extraction, invoked via its global. inference profile."
  type        = string
  default     = "anthropic.claude-haiku-4-5-20251001-v1:0"
}

# Models the app role may call so that P3 can measure them against the ones in
# use: `eval-curation --model` for chat, the retrieval check (PLAN.md § 8,
# PR 35) for embeddings. Every ID here is priced in pricing.yaml and called by
# the Smoke workflow, and tests/test_config.py checks both.

# Runtime catalog IDs, called through their `global.` inference profiles like
# chat_model_id. Claude only (CLAUDE.md #4). Sonnet 4.6: $3/$15 against Opus
# 4.6's $5/$25, and a 1024-token cache minimum against 4096, so the ~2k-token
# static prefix is cacheable on the first call. Sonnet 5 is cheaper still, but
# this account is refused the newest models (CLAUDE.md, Current state).
variable "chat_candidate_model_ids" {
  description = "Bedrock catalog model IDs of chat candidates (Claude only), invoked via their global. inference profiles."
  type        = list(string)
  default     = ["anthropic.claude-sonnet-4-6"]
}

# Foundation-model IDs invoked in-region (eu-west-1 serves both on demand), not
# through a profile: Titan has none, and in-region keeps one IAM grant per model.
variable "embed_candidate_model_ids" {
  description = "Bedrock foundation model IDs of embedding candidates, invoked in aws_region without an inference profile."
  type        = list(string)
  default = [
    "amazon.titan-embed-text-v2:0",
    "cohere.embed-v4:0",
  ]
}

# Catalog IDs, not Mantle IDs: model agreements are keyed on the Bedrock
# catalog ID, which for older models carries a date/version suffix that the
# Mantle ID drops (anthropic.claude-haiku-4-5 on Mantle). List them with
#   aws bedrock list-foundation-models --by-provider anthropic --query 'modelSummaries[].modelId'
variable "bedrock_agreement_model_ids" {
  description = "Bedrock catalog model IDs to enable (Marketplace agreement) — the models behind chat_model_id / extract_model_id, the Marketplace-sold candidates, plus models we need to test access for."
  type        = list(string)
  # Opus 4.6 (the chat model) is absent on purpose: the account can already
  # invoke it, so declaring an agreement here would fight whatever grants that.
  # Opus 5 and Sonnet 5 stay subscribed so nothing else has to change when
  # Bedrock's eligibility for the newest models is reassessed. Sonnet 4.6 and
  # Cohere Embed v4 are candidates sold through Marketplace; the smoke role
  # cannot subscribe on first use, so the agreement has to exist first. Titan
  # is Amazon's own and needs none.
  default = [
    "anthropic.claude-opus-5",
    "anthropic.claude-sonnet-5",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-sonnet-4-6",
    "cohere.embed-v4:0",
  ]
}
