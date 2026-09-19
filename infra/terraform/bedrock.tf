# The app's least-privilege Bedrock access. This is the first AWS resource
# in the project (CLAUDE.md #9). Nothing assumes this role yet — it's
# attached to compute in P7 (or assumed locally for now) — but the policy
# scope is fixed here rather than left to whichever principal shows up later.
#
# Scoped to only the configured chat/extract model ARNs, not `*`. Update
# `chat_model_id` / `extract_model_id` (and this list) when EMBED_MODEL is
# chosen in P3.
locals {
  bedrock_model_arns = [
    for model_id in [var.chat_model_id, var.extract_model_id] :
    "arn:${data.aws_partition.current.partition}:bedrock:${var.aws_region}::foundation-model/${model_id}"
  ]
}

data "aws_iam_policy_document" "app_bedrock_access" {
  statement {
    sid = "InvokeConfiguredModels"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
      # bedrock-mantle:CreateInference per CLAUDE.md #4/#9 (the Claude-native
      # Messages-API endpoint chat uses). Verify this action name in the
      # console alongside the open "confirm Opus 5 access" task in P0 —
      # Mantle's global-endpoint IAM shape wasn't in the provider docs used
      # to write this file.
      "bedrock-mantle:CreateInference",
    ]
    resources = local.bedrock_model_arns
  }
}

data "aws_iam_policy_document" "app_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    effect  = "Allow"
    principals {
      identifiers = [data.aws_caller_identity.current.account_id]
      type        = "AWS"
    }
  }
}

resource "aws_iam_role" "app" {
  name               = "art-curator-app"
  assume_role_policy = data.aws_iam_policy_document.app_assume_role.json
  description        = "Runtime role for The Art Curator. Assumed by compute from P7; assumable locally until then."
}

resource "aws_iam_role_policy" "app_bedrock_access" {
  name   = "bedrock-access"
  role   = aws_iam_role.app.id
  policy = data.aws_iam_policy_document.app_bedrock_access.json
}
