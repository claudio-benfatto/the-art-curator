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

variable "tf_state_bucket" {
  description = "Name of the hand-created S3 state bucket (infra/README.md step 1). No default — must match backend.tf's -backend-config bucket exactly, so the CI role's permissions target the right ARN."
  type        = string
}

# Keep these in sync with config.py's CHAT_MODEL / EXTRACT_MODEL defaults.
# EMBED_MODEL is chosen by measurement in P3 and added then.

# A bedrock-mantle model ID — matched against the bedrock-mantle:Model
# condition key in bedrock.tf, not used to build ARNs.
variable "chat_model_id" {
  description = "Bedrock model ID for chat (Claude only, per CLAUDE.md #4)."
  type        = string
  default     = "anthropic.claude-opus-5"
}

# A bedrock-runtime *catalog* ID. Extraction calls it through the global
# inference profile, so config.py's EXTRACT_MODEL is this with a `global.`
# prefix. Mantle refuses every model for this account (CLAUDE.md, Current
# state), so extraction runs on bedrock-runtime Converse until that's fixed.
variable "extract_model_id" {
  description = "Bedrock catalog model ID for extraction, invoked via its global. inference profile."
  type        = string
  default     = "anthropic.claude-haiku-4-5-20251001-v1:0"
}

# Catalog IDs, not Mantle IDs: model agreements are keyed on the Bedrock
# catalog ID, which for older models carries a date/version suffix that the
# Mantle ID drops (anthropic.claude-haiku-4-5 on Mantle). List them with
#   aws bedrock list-foundation-models --by-provider anthropic --query 'modelSummaries[].modelId'
variable "bedrock_agreement_model_ids" {
  description = "Bedrock catalog model IDs to enable (Marketplace agreement) — one per model behind chat_model_id / extract_model_id."
  type        = list(string)
  default = [
    "anthropic.claude-opus-5",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
  ]
}
