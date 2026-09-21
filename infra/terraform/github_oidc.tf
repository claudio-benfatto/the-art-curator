# Lets GitHub Actions assume AWS roles without a stored long-lived key
# (CLAUDE.md #10). Two roles, so write access is bound to the approval gate
# rather than to whoever can push a branch:
#
# - plan:  read-only. Assumable from PRs and pushes to main. A workflow edited
#          on a PR branch gets this role and nothing more.
# - apply: read-write. Trust is pinned to the `infra-apply` environment's OIDC
#          subject, so only a job that has passed that environment's required
#          reviewer can obtain it — a modified workflow on any branch still
#          stops at the approval prompt.
#
# thumbprint_list is omitted on purpose: AWS validates GitHub's certificate
# chain against its own trusted CAs, and fills in a thumbprint by itself when
# none is given. Setting `[]` makes every plan try to strip that value.
resource "aws_iam_openid_connect_provider" "github_actions" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
}

locals {
  partition      = data.aws_partition.current.partition
  account_id     = data.aws_caller_identity.current.account_id
  state_bucket   = "arn:${local.partition}:s3:::${var.tf_state_bucket}"
  state_object   = "${local.state_bucket}/art-curator/terraform.tfstate"
  state_lock     = "${local.state_bucket}/art-curator/terraform.tfstate.tflock"
  project_roles  = "arn:${local.partition}:iam::${local.account_id}:role/art-curator-*"
  oidc_provider  = "arn:${local.partition}:iam::${local.account_id}:oidc-provider/token.actions.githubusercontent.com"
  iam_read_roles = ["iam:GetRole", "iam:GetRolePolicy", "iam:ListRolePolicies", "iam:ListAttachedRolePolicies"]
}

data "aws_iam_policy_document" "github_assume_role" {
  for_each = {
    plan  = ["${var.github_oidc_sub_prefix}:pull_request", "${var.github_oidc_sub_prefix}:ref:refs/heads/main"]
    apply = ["${var.github_oidc_sub_prefix}:environment:${var.apply_environment}"]
  }

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
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = each.value
    }
  }
}

# --- plan: read-only ---------------------------------------------------------

resource "aws_iam_role" "github_plan" {
  name               = "art-curator-github-plan"
  assume_role_policy = data.aws_iam_policy_document.github_assume_role["plan"].json
  description        = "Read-only terraform plan from GitHub Actions (PRs, main) in ${var.github_repo} via OIDC."
}

data "aws_iam_policy_document" "github_plan" {
  statement {
    sid       = "StateBucketList"
    actions   = ["s3:ListBucket"]
    resources = [local.state_bucket]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["art-curator/*"]
    }
  }

  statement {
    sid       = "StateRead"
    actions   = ["s3:GetObject"]
    resources = [local.state_object, local.state_lock]
  }

  # plan takes the S3 lockfile too, so it never reads state mid-apply.
  # Writing the lock is the only write this role has.
  statement {
    sid       = "StateLock"
    actions   = ["s3:PutObject", "s3:DeleteObject"]
    resources = [local.state_lock]
  }

  statement {
    sid       = "ReadProjectIam"
    actions   = local.iam_read_roles
    resources = [local.project_roles]
  }

  statement {
    sid       = "ReadOidcProvider"
    actions   = ["iam:GetOpenIDConnectProvider"]
    resources = [local.oidc_provider]
  }
}

resource "aws_iam_role_policy" "github_plan" {
  name   = "terraform-plan"
  role   = aws_iam_role.github_plan.id
  policy = data.aws_iam_policy_document.github_plan.json
}

# --- apply: read-write, environment-gated ------------------------------------

resource "aws_iam_role" "github_apply" {
  name               = "art-curator-github-apply"
  assume_role_policy = data.aws_iam_policy_document.github_assume_role["apply"].json
  description        = "terraform apply from the ${var.apply_environment} environment in ${var.github_repo} via OIDC."
}

# Inline policies only — no Attach/DetachRolePolicy, so this role can't hang
# an AWS-managed policy (e.g. AdministratorAccess) off an art-curator-* role.
data "aws_iam_policy_document" "github_apply" {
  statement {
    sid       = "StateBucketList"
    actions   = ["s3:ListBucket"]
    resources = [local.state_bucket]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["art-curator/*"]
    }
  }

  statement {
    sid       = "StateObjects"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = [local.state_object, local.state_lock]
  }

  statement {
    sid = "ManageProjectIamRoles"
    actions = concat(local.iam_read_roles, [
      "iam:CreateRole",
      "iam:DeleteRole",
      "iam:UpdateRole",
      "iam:UpdateRoleDescription",
      "iam:UpdateAssumeRolePolicy",
      "iam:TagRole",
      "iam:UntagRole",
      "iam:PutRolePolicy",
      "iam:DeleteRolePolicy",
      "iam:ListInstanceProfilesForRole",
    ])
    resources = [local.project_roles]
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
    resources = [local.oidc_provider]
  }
}

resource "aws_iam_role_policy" "github_apply" {
  name   = "terraform-apply"
  role   = aws_iam_role.github_apply.id
  policy = data.aws_iam_policy_document.github_apply.json
}
