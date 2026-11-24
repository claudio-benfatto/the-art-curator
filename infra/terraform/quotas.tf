# Bedrock quotas the app needs above the account's defaults.
#
# Opus 4.6 (chat, and the judge by default) is allowed 5 requests a minute on
# this account (2026-11-23), against 3,000,000 tokens a minute: one
# conversation's turn is up to 5 calls (4 tool rounds and a final reply), so a
# single user can be refused mid-turn, and eval-curation is refused after its
# first scenario. The eval paces itself to REQUESTS_PER_MINUTE meanwhile
# (llm/client.RequestPacer); raise that setting to what AWS grants.
#
# A `value` above the current limit files an increase request with AWS, which
# may take days or be granted in part. While it is pending, Terraform shows the
# requested value. If AWS grants less than asked, lower `value` to what was
# granted: otherwise every apply asks again. `terraform destroy` only drops the
# resource from state; quotas are never lowered.
#
# Chat runs on the global inference profile, so the global quota is the one
# that counts, not the cross-region (geographic) one of the same name.

locals {
  # Must name chat_model_id's quota exactly: the data source looks it up by name.
  chat_requests_quota_name = "Global cross-region model inference requests per minute for Anthropic Claude Opus 4.6 V1"
}

data "aws_servicequotas_service_quota" "chat_requests_per_minute" {
  service_code = "bedrock"
  quota_name   = local.chat_requests_quota_name
}

resource "aws_servicequotas_service_quota" "chat_requests_per_minute" {
  service_code = "bedrock"
  quota_code   = data.aws_servicequotas_service_quota.chat_requests_per_minute.quota_code
  value        = var.chat_requests_per_minute
}
