# The bucket itself can't be declared here — Terraform can't create the
# backend it's about to store its own state in. It's created by hand once,
# per infra/README.md, then this block is completed at init time:
#
#   terraform init -backend-config="bucket=<state bucket name>"
#
# S3-native locking (`use_lockfile`) needs no DynamoDB table.
terraform {
  backend "s3" {
    key          = "art-curator/terraform.tfstate"
    region       = "eu-west-1"
    use_lockfile = true
    encrypt      = true
  }
}
