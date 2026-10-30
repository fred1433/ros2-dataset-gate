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
  description = "Worker image URI (built from docker/worker.Dockerfile)."
  type        = string
}

variable "max_vcpus" {
  type    = number
  default = 256
}
