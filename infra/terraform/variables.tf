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
# These are bedrock-mantle model IDs — matched against the bedrock-mantle:Model
# condition key in bedrock.tf, not used to build ARNs.
# EMBED_MODEL is chosen by measurement in P3 and added to this list then.
variable "chat_model_id" {
  description = "Bedrock model ID for chat (Claude only, per CLAUDE.md #4)."
  type        = string
  default     = "anthropic.claude-opus-5"
}

variable "extract_model_id" {
  description = "Bedrock model ID for extraction."
  type        = string
  default     = "anthropic.claude-haiku-4-5"
}
