# The app's least-privilege Bedrock access. This is the first AWS resource
# in the project (CLAUDE.md #9). Nothing assumes this role yet — it's
# attached to compute in P7 (or assumed locally for now) — but the policy
# scope is fixed here rather than left to whichever principal shows up later.
#
# Claude goes through the bedrock-mantle endpoint (AnthropicBedrockMantle,
# CLAUDE.md #4). Mantle authorizes CreateInference against a Mantle *project*
# resource, not a bedrock foundation-model ARN, and narrows to specific
# models via the `bedrock-mantle:Model` condition key — so the scope lives in
# the condition, keyed on the same Mantle model IDs config.py uses.
#
# No bedrock:InvokeModel yet: nothing calls bedrock-runtime until a
# non-Claude model is chosen (EMBED_MODEL, P3). When it is, add the global
# cross-region inference statements from the Bedrock user guide
# (inference-profile ARN + regional and region-less foundation-model ARNs,
# conditioned on bedrock:InferenceProfileArn) — runtime model IDs differ from
# Mantle's (e.g. anthropic.claude-haiku-4-5-20251001-v1:0).
data "aws_iam_policy_document" "app_bedrock_access" {
  statement {
    sid     = "MantleInvokeConfiguredModels"
    actions = ["bedrock-mantle:CreateInference"]
    resources = [
      "arn:${data.aws_partition.current.partition}:bedrock-mantle:${var.aws_region}:${data.aws_caller_identity.current.account_id}:project/*",
    ]
    condition {
      test     = "StringEquals"
      variable = "bedrock-mantle:Model"
      values   = [var.chat_model_id, var.extract_model_id]
    }
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

# Model enablement (replaces the old console "Model access" page). Third-party
# models are subscribed through AWS Marketplace on first use; declaring the
# agreement here makes that explicit and reviewable instead of a side effect
# of whichever principal calls first — and means the app role never needs
# aws-marketplace:* permissions. Creating an agreement accepts the model's
# EULA (https://aws.amazon.com/legal/bedrock/third-party-models/).
#
# The Anthropic first-time-use form (aws_bedrock_use_case_for_model_access)
# is not needed: it doesn't apply to models called through bedrock-mantle.
data "aws_bedrock_foundation_model_agreement_offers" "enabled" {
  for_each = toset(var.bedrock_agreement_model_ids)
  model_id = each.value
}

resource "aws_bedrock_foundation_model_agreement" "enabled" {
  for_each    = toset(var.bedrock_agreement_model_ids)
  model_id    = each.value
  offer_token = data.aws_bedrock_foundation_model_agreement_offers.enabled[each.value].offers[0].offer_token

  # Offer tokens are reissued over time. A new token must not destroy and
  # recreate a working agreement.
  lifecycle {
    ignore_changes = [offer_token]
  }
}
