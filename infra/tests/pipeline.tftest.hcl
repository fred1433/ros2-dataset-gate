# Offline: the AWS provider is mocked, nothing is created, no credentials.
# This checks the configuration's own logic, not IAM or service behaviour.

mock_provider "aws" {
  override_during = plan
  mock_resource "aws_s3_bucket" {
    defaults = {
      arn = "arn:aws:s3:::dsgate-datasets"
    }
  }
  mock_resource "aws_iam_role" {
    defaults = {
      arn = "arn:aws:iam::123456789012:role/mocked"
    }
  }
  mock_resource "aws_iam_instance_profile" {
    defaults = {
      arn = "arn:aws:iam::123456789012:instance-profile/mocked"
    }
  }
}

variables {
  subnet_ids         = ["subnet-00000000000000001"]
  security_group_ids = ["sg-00000000000000001"]
  worker_image       = "123456789012.dkr.ecr.us-west-2.amazonaws.com/dsgate-worker:0.1.0"
}

run "plan_offline" {
  command = plan

  assert {
    condition     = aws_s3_bucket_versioning.data.versioning_configuration[0].status == "Enabled"
    error_message = "raw logs must be versioned: submissions pin object versions"
  }

  assert {
    condition     = aws_batch_job_definition.worker.retry_strategy[0].attempts == 3
    error_message = "worker retries: 3 attempts"
  }

  assert {
    condition     = lower(aws_batch_job_definition.worker.retry_strategy[0].evaluate_on_exit[0].action) == "retry" && aws_batch_job_definition.worker.retry_strategy[0].evaluate_on_exit[0].on_status_reason == "Host EC2*"
    error_message = "only Spot reclamation is retried by Batch"
  }

  assert {
    condition     = lower(aws_batch_job_definition.worker.retry_strategy[0].evaluate_on_exit[1].action) == "exit"
    error_message = "every other failure exits and is resumed explicitly"
  }

  assert {
    condition     = jsondecode(aws_batch_job_definition.worker.container_properties).command == ["python", "-m", "dsgate.worker"]
    error_message = "job definition runs the worker module"
  }

  assert {
    condition     = aws_batch_compute_environment.spot.compute_resources[0].allocation_strategy == "SPOT_PRICE_CAPACITY_OPTIMIZED"
    error_message = "Spot with price-capacity-optimized allocation"
  }

  assert {
    condition = alltrue([
      for s in jsondecode(aws_iam_role_policy.worker_s3.policy).Statement :
      !contains(s.Action, "s3:DeleteObject") && !contains(s.Action, "s3:*")
    ])
    error_message = "workers never delete or hold s3:*"
  }

  assert {
    condition = alltrue([
      for r in one([for s in jsondecode(aws_iam_role_policy.worker_s3.policy).Statement : s if s.Sid == "WriteImmutableOutputs"]).Resource :
      endswith(r, "/cas/*") || endswith(r, "/receipts/*")
    ])
    error_message = "workers write only content-addressed outputs and receipts"
  }
}
