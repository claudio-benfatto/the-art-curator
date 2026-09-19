output "github_actions_role_arn" {
  description = "Set as the GitHub repo/environment variable AWS_ROLE_ARN so CI can authenticate via OIDC."
  value       = aws_iam_role.github_actions.arn
}

output "app_role_arn" {
  description = "Runtime role with Bedrock access. Referenced by compute config from P7 onward."
  value       = aws_iam_role.app.arn
}
