# Lets GitHub Actions assume an AWS role without a stored long-lived key
# (CLAUDE.md #10). One role, usable from any workflow run in this repo
# (`repo:<org>/<name>:*`) — `plan` runs on PRs, `apply` runs only from
# `main` behind a protected environment that needs manual approval, so the
# environment gate is what limits write access, not a second role.
#
# No thumbprint_list: AWS validates GitHub's OIDC certificate chain against
# its own trusted CAs for this provider (see aws_iam_openid_connect_provider
# docs) — the previously-required SHA1 thumbprint is not needed.
resource "aws_iam_openid_connect_provider" "github_actions" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = []
}

data "aws_iam_policy_document" "github_actions_assume_role" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    effect  = "Allow"

    principals {
      identifiers = [aws_iam_openid_connect_provider.github_actions.arn]
      type        = "Federated"
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repo}:*"]
    }
  }
}

resource "aws_iam_role" "github_actions" {
  name               = "art-curator-github-actions"
  assume_role_policy = data.aws_iam_policy_document.github_actions_assume_role.json
  description        = "Assumed by GitHub Actions in ${var.github_repo} via OIDC. No long-lived AWS keys in repo secrets."
}

# Scoped to managing only this project's own resources (name/ARN prefixed
# `art-curator-`) plus the state bucket it reads/writes on every plan and
# apply. Not scoped down further into separate plan/apply tiers — see
# github_oidc.tf's top comment.
data "aws_iam_policy_document" "github_actions_terraform" {
  statement {
    sid       = "StateBucketList"
    actions   = ["s3:ListBucket"]
    resources = ["arn:${data.aws_partition.current.partition}:s3:::${var.tf_state_bucket}"]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["art-curator/*"]
    }
  }

  statement {
    sid = "StateObjects"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = [
      "arn:${data.aws_partition.current.partition}:s3:::${var.tf_state_bucket}/art-curator/terraform.tfstate",
      "arn:${data.aws_partition.current.partition}:s3:::${var.tf_state_bucket}/art-curator/terraform.tfstate.tflock",
    ]
  }

  statement {
    sid = "ManageProjectIamRolesAndPolicies"
    actions = [
      "iam:CreateRole",
      "iam:DeleteRole",
      "iam:GetRole",
      "iam:UpdateRole",
      "iam:UpdateAssumeRolePolicy",
      "iam:TagRole",
      "iam:UntagRole",
      "iam:PutRolePolicy",
      "iam:GetRolePolicy",
      "iam:DeleteRolePolicy",
      "iam:ListRolePolicies",
      "iam:ListAttachedRolePolicies",
      "iam:AttachRolePolicy",
      "iam:DetachRolePolicy",
    ]
    resources = [
      "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:role/art-curator-*",
    ]
  }

  statement {
    sid = "ManageProjectOidcProvider"
    actions = [
      "iam:CreateOpenIDConnectProvider",
      "iam:DeleteOpenIDConnectProvider",
      "iam:GetOpenIDConnectProvider",
      "iam:UpdateOpenIDConnectProviderThumbprint",
      "iam:AddClientIDToOpenIDConnectProvider",
      "iam:RemoveClientIDFromOpenIDConnectProvider",
      "iam:TagOpenIDConnectProvider",
      "iam:UntagOpenIDConnectProvider",
    ]
    resources = [
      "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/token.actions.githubusercontent.com",
    ]
  }
}

resource "aws_iam_role_policy" "github_actions_terraform" {
  name   = "terraform-plan-apply"
  role   = aws_iam_role.github_actions.id
  policy = data.aws_iam_policy_document.github_actions_terraform.json
}
