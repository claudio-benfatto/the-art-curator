# Infra — manual steps

Everything AWS lives in `infra/terraform/` (CLAUDE.md #9). This file is only
for the handful of steps Terraform genuinely can't express — bootstrapping
the thing Terraform itself depends on, and one-time account settings with no
Terraform resource.

Do these once, in order, with your own AWS credentials (`aws configure` or
`aws sso login`). After step 5, everything else runs through CI.

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

## 2. Bedrock model access — nothing to do by hand

There is no console "Model access" step any more. Models are enabled by an
AWS Marketplace agreement, which Terraform declares
(`aws_bedrock_foundation_model_agreement` in `bedrock.tf`, one per ID in
`bedrock_agreement_model_ids`). Applying it accepts each model's EULA
(https://aws.amazon.com/legal/bedrock/third-party-models/).

- The account needs a valid payment method for Marketplace purchases.
- The Anthropic first-time-use form is declared too
  (`aws_bedrock_use_case_for_model_access`). Mantle calls don't need it, but
  creating some agreements does (Haiku 4.5 was refused without it). It is
  one-time per account and can't be updated or deleted once submitted.
- Agreement IDs are **catalog** IDs (`anthropic.claude-haiku-4-5-20251001-v1:0`),
  not the Mantle IDs in `chat_model_id` / `extract_model_id`. Check access
  with `aws bedrock get-foundation-model-availability --model-id <catalog id>`
  — `agreementAvailability.status` should be `AVAILABLE`.

The app role's policy allows `bedrock-mantle:CreateInference` only for the
chat model's Mantle ID (`bedrock-mantle:Model` condition), and
`bedrock:InvokeModel` only for the extraction model through its `global.`
inference profile. An IAM denial names the denied action and context — fix the
variable, not the policy scope.

**Availability is not access.** `get-foundation-model-availability` can report
`AUTHORIZED` / `AVAILABLE` while calls are still refused with "not available
for this account" — a 403 with no IAM action in the message is account-level,
not IAM. As of 2026-09-22 that is the case for Opus 5 (every path) and for
every model on Mantle; see CLAUDE.md § Current state.

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

The apply created two CI roles (`github_oidc.tf`): a read-only **plan** role
for PRs and `main`, and an **apply** role that trusts only the `infra-apply`
environment's OIDC subject. Set both ARNs from the apply output, plus the
region and bucket name the workflow needs to reconstruct the backend config:

```bash
gh variable set AWS_PLAN_ROLE_ARN  --body "$(terraform output -raw github_plan_role_arn)"
gh variable set AWS_APPLY_ROLE_ARN --body "$(terraform output -raw github_apply_role_arn)"
gh variable set AWS_REGION         --body "eu-west-1"
gh variable set TF_STATE_BUCKET    --body "<bucket-name>"
```

Both roles trust GitHub's OIDC `sub` claim exactly. This repo uses GitHub's
immutable-subject format (`repo:<owner>@<id>/<repo>@<id>:…`), held in
`github_oidc_sub_prefix` (`variables.tf`). If CI fails with "Not authorized to
perform sts:AssumeRoleWithWebIdentity", compare that variable against:

```bash
gh api repos/<owner>/<repo>/actions/oidc/customization/sub --jq .sub_claim_prefix
```

Then create the protected `infra-apply` environment (Settings → Environments):

- **Required reviewers:** yourself. This is the control that grants write
  access to AWS — anyone who can push a branch can edit a workflow, but only
  a job that passes this approval can assume the apply role (CLAUDE.md #10).
- **Deployment branches:** `main` only, so an approval prompt can't be raised
  from a feature branch in the first place.

## 5. Wire up the smoke job

The apply also created a **smoke** role (`github_oidc.tf`) with the app's
Bedrock policy, trusting only the `smoke` environment's OIDC subject. It backs
`.github/workflows/smoke.yml`, the one workflow that calls a model.

```bash
gh variable set AWS_SMOKE_ROLE_ARN --body "$(terraform output -raw github_smoke_role_arn)"
```

Create the `smoke` environment (Settings → Environments) with **Deployment
branches:** `main` only. No required reviewer: the workflow is manual
(`workflow_dispatch`), which is the deliberate step, and a run costs a
fraction of a cent. Run it from Actions → Smoke → Run workflow, or
`gh workflow run smoke.yml`.

## After that

Every PR that touches `infra/` gets `fmt`, `validate` and `plan` from CI
automatically. Merges to `main` run `apply`, gated on that approval. Nothing
here should need a console click or an ad-hoc `aws` CLI call again — if it
does, that's drift; fix it by importing the resource into Terraform, not by
continuing to click.
