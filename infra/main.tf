# Minimal infrastructure for the curation workers and the training role.
# Checked offline with `terraform test` and a mocked AWS provider only:
# IAM, networking and service integration remain to be verified on AWS.

locals {
  bucket = "${var.name}-datasets"
  # Prefixes the workers may write. Raw logs are read-only for them.
  worker_write_prefixes = ["cas/*", "receipts/*"]
}

# ------------------------------------------------------------------ storage

resource "aws_s3_bucket" "data" {
  bucket = local.bucket
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled" # submissions pin object versions of raw logs
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    id     = "abort-incomplete-multipart"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload {
      days_after_initiation = 2
    }
  }
}

resource "aws_ecr_repository" "worker" {
  name                 = "${var.name}-worker"
  image_tag_mutability = "IMMUTABLE"
}

# --------------------------------------------------------------------- IAM

resource "aws_iam_role" "worker_job" {
  name = "${var.name}-worker-job"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "worker_s3" {
  name = "${var.name}-worker-s3"
  role = aws_iam_role.worker_job.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ReadInputs"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:GetObjectVersion"]
        Resource = ["${aws_s3_bucket.data.arn}/*"]
      },
      {
        Sid      = "WriteImmutableOutputs"
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = [for p in local.worker_write_prefixes : "${aws_s3_bucket.data.arn}/${p}"]
      },
      {
        Sid      = "ListForResume"
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = [aws_s3_bucket.data.arn]
      }
    ]
  })
}

resource "aws_iam_role" "batch_instance" {
  name = "${var.name}-batch-instance"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "batch_instance_ecs" {
  role       = aws_iam_role.batch_instance.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_instance_profile" "batch_instance" {
  name = "${var.name}-batch-instance"
  role = aws_iam_role.batch_instance.name
}

resource "aws_iam_role" "sagemaker_training" {
  name = "${var.name}-sagemaker-training"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "sagemaker_read_versions" {
  name = "${var.name}-sagemaker-read"
  role = aws_iam_role.sagemaker_training.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid      = "ReadPublishedVersionsAndShards"
      Effect   = "Allow"
      Action   = ["s3:GetObject"]
      Resource = ["${aws_s3_bucket.data.arn}/versions/*", "${aws_s3_bucket.data.arn}/cas/*"]
    }]
  })
}

# ------------------------------------------------------------------- Batch

resource "aws_batch_compute_environment" "spot" {
  name = "${var.name}-spot"
  type = "MANAGED"

  compute_resources {
    type                = "SPOT"
    allocation_strategy = "SPOT_PRICE_CAPACITY_OPTIMIZED"
    min_vcpus           = 0
    max_vcpus           = var.max_vcpus
    instance_type       = ["optimal"]
    subnets             = var.subnet_ids
    security_group_ids  = var.security_group_ids
    instance_role       = aws_iam_instance_profile.batch_instance.arn
  }
}

resource "aws_batch_job_queue" "curation" {
  name     = "${var.name}-curation"
  state    = "ENABLED"
  priority = 1

  compute_environment_order {
    order               = 1
    compute_environment = aws_batch_compute_environment.spot.arn
  }
}

resource "aws_batch_job_definition" "worker" {
  name = "${var.name}-worker"
  type = "container"

  # A Spot reclamation is retried by Batch. Anything else (a bug, an integrity
  # failure, a conflict) exits non-zero once and is resumed explicitly from
  # the frozen submission. Data defects never reach here: they exit 0 with a
  # quarantined receipt.
  retry_strategy {
    attempts = 3
    evaluate_on_exit {
      on_status_reason = "Host EC2*"
      action           = "RETRY"
    }
    evaluate_on_exit {
      on_reason = "*"
      action    = "EXIT"
    }
  }

  timeout {
    attempt_duration_seconds = 3600
  }

  container_properties = jsonencode({
    image      = var.worker_image
    command    = ["python", "-m", "dsgate.worker"]
    jobRoleArn = aws_iam_role.worker_job.arn
    resourceRequirements = [
      { type = "VCPU", value = "2" },
      { type = "MEMORY", value = "4096" }
    ]
    environment = [{ name = "DSGATE_BUCKET", value = local.bucket }]
  })
}
