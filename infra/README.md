# Infra — manual steps

Everything AWS lives in `infra/terraform/` (CLAUDE.md #9). This file is only
for the handful of steps Terraform genuinely can't express — bootstrapping
the thing Terraform itself depends on, and one-time account settings with no
Terraform resource.

Do these once, in order, with your own AWS credentials (`aws configure` or
`aws sso login`). After step 4, everything else runs through CI.

## 1. Create the state bucket

Terraform can't create the bucket that holds its own state. Pick a globally
unique name (e.g. `art-curator-tfstate-<your account id>`) and:

```bash
aws s3api create-bucket \
  --bucket <bucket-name> \
  --region eu-west-1 \
  --create-bucket-configuration LocationConstraint=eu-west-1

aws s3api put-bucket-versioning \
  --bucket <bucket-name> \
  --versioning-configuration Status=Enabled

aws s3api put-bucket-encryption \
  --bucket <bucket-name> \
  --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'

aws s3api put-public-access-block \
  --bucket <bucket-name> \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

## 2. Accept Bedrock model access terms

Bedrock model access is opt-in per model, per account, and has no Terraform
resource. In the AWS Console: **Bedrock → Model access → Manage model
access**, request access to the Claude models named in
`infra/terraform/variables.tf` (`chat_model_id`, `extract_model_id`). This is
also the "confirm Opus 5 access" check from PLAN.md's P0 open risks — Opus 5
isn't open to every account.

While there, sanity-check the IAM action name in `bedrock.tf`
(`bedrock-mantle:CreateInference`) against what the console/IAM policy
visual editor shows for the Mantle endpoint — that part of the policy was
written from the AWS provider's general Bedrock docs, not Mantle-specific
ones, and Mantle is new enough that this is worth a second look.

## 3. First apply (local credentials)

The GitHub Actions role has to exist before GitHub Actions can use it —
so the first `apply` runs from your machine:

```bash
cd infra/terraform
terraform init -backend-config="bucket=<bucket-name>"
terraform apply -var="tf_state_bucket=<bucket-name>"
```

Keep `-var="tf_state_bucket=..."` (or a `*.tfvars` file, gitignored) for
every future local run too — it's how the CI role's policy knows which
bucket ARN to scope itself to.

## 4. Wire up CI

Take `github_actions_role_arn` from the apply output and set it as a repo
(or environment) variable, plus the region and bucket name the workflow
needs to reconstruct the backend config:

```bash
gh variable set AWS_ROLE_ARN --body "<github_actions_role_arn output>"
gh variable set AWS_REGION --body "eu-west-1"
gh variable set TF_STATE_BUCKET --body "<bucket-name>"
```

Then create the protected `infra-apply` environment (Settings → Environments)
with yourself as a required reviewer, so `terraform apply` on `main` waits
for manual approval (CLAUDE.md #10).

## After that

Every PR that touches `infra/` gets `fmt`, `validate` and `plan` from CI
automatically. Merges to `main` run `apply`, gated on that approval. Nothing
here should need a console click or an ad-hoc `aws` CLI call again — if it
does, that's drift; fix it by importing the resource into Terraform, not by
continuing to click.
