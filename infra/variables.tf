variable "region" {
  type    = string
  default = "us-west-2"
}

variable "name" {
  description = "Prefix for every resource."
  type        = string
  default     = "dsgate"
}

variable "subnet_ids" {
  description = "Private subnets for the Batch compute environment."
  type        = list(string)
}

variable "security_group_ids" {
  type = list(string)
}

variable "worker_image" {
  description = "Worker image pinned by digest (repo@sha256:...), built from docker/worker.Dockerfile."
  type        = string
  validation {
    condition     = can(regex("@sha256:[0-9a-f]{64}$", var.worker_image))
    error_message = "worker_image must be pinned by digest (...@sha256:<64 hex>), not a mutable tag."
  }
}

variable "max_vcpus" {
  type    = number
  default = 256
}
