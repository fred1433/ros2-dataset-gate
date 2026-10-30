terraform {
  required_version = ">= 1.11.0" # terraform test uses override_during (1.11)
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.region
}
