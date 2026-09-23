output "github_plan_role_arn" {
  description = "Set as the GitHub repo variable AWS_PLAN_ROLE_ARN. Read-only; used by the plan job."
  value       = aws_iam_role.github_plan.arn
}

output "github_apply_role_arn" {
  description = "Set as the GitHub repo variable AWS_APPLY_ROLE_ARN. Only assumable from the infra-apply environment."
  value       = aws_iam_role.github_apply.arn
}

output "github_smoke_role_arn" {
  description = "Set as the GitHub repo variable AWS_SMOKE_ROLE_ARN. Only assumable from the smoke environment."
  value       = aws_iam_role.github_smoke.arn
}

output "app_role_arn" {
  description = "Runtime role with Bedrock access. Referenced by compute config from P7 onward."
  value       = aws_iam_role.app.arn
}
