provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      project    = "art-curator"
      managed_by = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}

data "aws_partition" "current" {}

data "aws_region" "current" {}
