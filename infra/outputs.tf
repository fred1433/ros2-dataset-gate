output "bucket" {
  value = aws_s3_bucket.data.bucket
}

output "job_queue" {
  value = aws_batch_job_queue.curation.name
}

output "job_definition" {
  value = aws_batch_job_definition.worker.name
}

output "sagemaker_role_arn" {
  value = aws_iam_role.sagemaker_training.arn
}
