variable "project_id" {
  description = "GCP project that owns the platform resources."
  type        = string
}

variable "region" {
  description = "GCP region for the regional GKE cluster."
  type        = string
  default     = "europe-west3"
}

variable "cluster_name" {
  description = "Name of the regional GKE cluster."
  type        = string
  default     = "graphrag-prod"
}

variable "node_machine_type" {
  description = "Machine type for the application node pool."
  type        = string
  default     = "e2-standard-4"
}

variable "min_nodes" {
  description = "Minimum number of application nodes across the regional pool."
  type        = number
  default     = 3
}

variable "max_nodes" {
  description = "Maximum number of application nodes across the regional pool."
  type        = number
  default     = 9
}

variable "master_ipv4_cidr_block" {
  description = "RFC1918 /28 for the private GKE control plane peering range."
  type        = string
  default     = "172.16.0.32/28"
}

variable "master_authorized_networks" {
  description = "CIDRs allowed to reach the GKE control plane endpoint (e.g. CI runners, operator VPN). Empty = no public access."
  type = list(object({
    cidr_block   = string
    display_name = string
  }))
  default = []
}

variable "backup_retention_days" {
  description = "Days to retain tenant graph backup objects in Cloud Storage."
  type        = number
  default     = 35
}
