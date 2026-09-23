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
# Extraction goes through bedrock-runtime Converse on a global cross-region
# inference profile. That takes the three-part grant from the Bedrock user
# guide (global-cross-region-inference.html): the profile in the requesting
# region, the model in that region, and the region-less model ARN that global
# routing evaluates with aws:RequestedRegion = "unspecified". Both model grants
# are pinned to the profile via bedrock:InferenceProfileArn, so the role can't
# call the model directly or through some other profile. Converse is
# authorized as bedrock:InvokeModel.
locals {
  extract_profile_arn = "arn:${local.partition}:bedrock:${var.aws_region}:${local.account_id}:inference-profile/global.${var.extract_model_id}"
}

data "aws_iam_policy_document" "app_bedrock_access" {
  statement {
    sid     = "MantleInvokeConfiguredModels"
    actions = ["bedrock-mantle:CreateInference"]
    resources = [
      "arn:${local.partition}:bedrock-mantle:${var.aws_region}:${local.account_id}:project/*",
    ]
    condition {
      test     = "StringEquals"
      variable = "bedrock-mantle:Model"
      values   = [var.chat_model_id]
    }
  }

  statement {
    sid       = "RuntimeExtractProfile"
    actions   = ["bedrock:InvokeModel"]
    resources = [local.extract_profile_arn]
    condition {
      test     = "StringEquals"
      variable = "aws:RequestedRegion"
      values   = [var.aws_region]
    }
  }

  statement {
    sid       = "RuntimeExtractRegionalModel"
    actions   = ["bedrock:InvokeModel"]
    resources = ["arn:${local.partition}:bedrock:${var.aws_region}::foundation-model/${var.extract_model_id}"]
    condition {
      test     = "StringEquals"
      variable = "aws:RequestedRegion"
      values   = [var.aws_region]
    }
    condition {
      test     = "StringEquals"
      variable = "bedrock:InferenceProfileArn"
      values   = [local.extract_profile_arn]
    }
  }

  statement {
    sid       = "RuntimeExtractGlobalModel"
    actions   = ["bedrock:InvokeModel"]
    resources = ["arn:${local.partition}:bedrock:::foundation-model/${var.extract_model_id}"]
    condition {
      test     = "StringEquals"
      variable = "aws:RequestedRegion"
      values   = ["unspecified"]
    }
    condition {
      test     = "StringEquals"
      variable = "bedrock:InferenceProfileArn"
      values   = [local.extract_profile_arn]
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
# The Anthropic first-time-use (FTU) form below is a prerequisite for
# *creating the agreement* for some models (Haiku 4.5 was refused without
# it), even though Mantle calls themselves don't need it. It is one-time per
# account, cannot be updated or deleted once submitted, and `terraform
# destroy` only drops it from state. Its content is a statement to Anthropic
# and AWS — change it only with the account owner's say-so.
resource "aws_bedrock_use_case_for_model_access" "anthropic" {
  form_data = jsonencode({
    companyName         = "Claudio Benfatto (individual developer)"
    companyWebsite      = "https://github.com/claudio-benfatto"
    intendedUsers       = "0" # internal
    industryOption      = "Arts and entertainment"
    otherIndustryOption = ""
    useCases            = "Proof-of-concept art curator for Catalunya: an assistant that recommends current exhibitions and events. Claude writes original short summaries from public venue listings and answers users' questions in chat with recommended itineraries. No personal data beyond chat messages; outputs are original text, not reproductions of source material."
  })

  lifecycle {
    # Updates are unsupported by the API; a drifted form must not trigger a
    # replacement attempt.
    ignore_changes = [form_data]
  }
}

data "aws_bedrock_foundation_model_agreement_offers" "enabled" {
  for_each = toset(var.bedrock_agreement_model_ids)
  model_id = each.value
}

resource "aws_bedrock_foundation_model_agreement" "enabled" {
  for_each    = toset(var.bedrock_agreement_model_ids)
  model_id    = each.value
  offer_token = data.aws_bedrock_foundation_model_agreement_offers.enabled[each.value].offers[0].offer_token

  depends_on = [aws_bedrock_use_case_for_model_access.anthropic]

  # Offer tokens are reissued over time. A new token must not destroy and
  # recreate a working agreement.
  lifecycle {
    ignore_changes = [offer_token]
  }
}
