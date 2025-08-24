
terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.80"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
  }
}

provider "google" {
  project = var.gcp_project
  region  = var.gcp_region
}

variable "gcp_project" {
  description = "GCP project ID"
  type        = string
}

variable "gcp_region" {
  description = "GCP region for deployment"
  type        = string
}


# GKE Cluster: nautilus-primary-eu-west
resource "google_container_cluster" "nautilus_primary_eu_west" {
  name     = "nautilus-primary-eu-west"
  location = var.gcp_region
  
  # Network configuration
  network    = google_compute_network.nautilus_primary_eu_west.self_link
  subnetwork = google_compute_subnetwork.nautilus_primary_eu_west_private.self_link

  # Cluster configuration
  initial_node_count       = 1
  remove_default_node_pool = true
  min_master_version       = "1.28"

  # Private cluster configuration
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"

    master_global_access_config {
      enabled = true
    }
  }

  # IP allocation for VPC-native
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Network policy
  network_policy {
    enabled  = true
    provider = "CALICO"
  }

  # Workload identity
  workload_identity_config {
    workload_pool = "{var.gcp_project}.svc.id.goog"
  }

  # Logging and monitoring
  logging_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "WORKLOADS",
      "API_SERVER"
    ]
  }

  monitoring_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "WORKLOADS"
    ]
  }
}

# VPC Network
resource "google_compute_network" "nautilus_primary_eu_west" {
  name                    = "nautilus-primary-eu-west-vpc"
  auto_create_subnetworks = false
}

# Private Subnet  
resource "google_compute_subnetwork" "nautilus_primary_eu_west_private" {
  name          = "nautilus-primary-eu-west-private-subnet"
  ip_cidr_range = "10.3.0.0/16"
  region        = var.gcp_region
  network       = google_compute_network.nautilus_primary_eu_west.self_link

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.13.0.0/14"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.23.0.0/16"
  }

  private_ip_google_access = true
}

# Node Pool: ultra-low-latency
resource "google_container_node_pool" "nautilus_primary_eu_west_pool_0" {
  name       = "ultra-low-latency"
  cluster    = google_container_cluster.nautilus_primary_eu_west.name
  location   = var.gcp_region
  node_count = 4

  autoscaling {
    min_node_count = 4
    max_node_count = 12
  }

  node_config {
    machine_type = "c2-standard-8"
    disk_size_gb = 200
    disk_type    = "pd-ssd"

    # Workload identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      tier = "primary-trading"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}


# GKE Cluster: nautilus-hub-eu-central
resource "google_container_cluster" "nautilus_hub_eu_central" {
  name     = "nautilus-hub-eu-central"
  location = var.gcp_region
  
  # Network configuration
  network    = google_compute_network.nautilus_hub_eu_central.self_link
  subnetwork = google_compute_subnetwork.nautilus_hub_eu_central_private.self_link

  # Cluster configuration
  initial_node_count       = 1
  remove_default_node_pool = true
  min_master_version       = "1.28"

  # Private cluster configuration
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"

    master_global_access_config {
      enabled = true
    }
  }

  # IP allocation for VPC-native
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Network policy
  network_policy {
    enabled  = true
    provider = "CALICO"
  }

  # Workload identity
  workload_identity_config {
    workload_pool = "{var.gcp_project}.svc.id.goog"
  }

  # Logging and monitoring
  logging_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "WORKLOADS",
      "API_SERVER"
    ]
  }

  monitoring_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "WORKLOADS"
    ]
  }
}

# VPC Network
resource "google_compute_network" "nautilus_hub_eu_central" {
  name                    = "nautilus-hub-eu-central-vpc"
  auto_create_subnetworks = false
}

# Private Subnet  
resource "google_compute_subnetwork" "nautilus_hub_eu_central_private" {
  name          = "nautilus-hub-eu-central-private-subnet"
  ip_cidr_range = "10.4.0.0/16"
  region        = var.gcp_region
  network       = google_compute_network.nautilus_hub_eu_central.self_link

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.14.0.0/14"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.24.0.0/16"
  }

  private_ip_google_access = true
}

# Node Pool: high-performance
resource "google_container_node_pool" "nautilus_hub_eu_central_pool_0" {
  name       = "high-performance"
  cluster    = google_container_cluster.nautilus_hub_eu_central.name
  location   = var.gcp_region
  node_count = 3

  autoscaling {
    min_node_count = 3
    max_node_count = 8
  }

  node_config {
    machine_type = "c2-standard-4"
    disk_size_gb = 50
    disk_type    = "pd-ssd"

    # Workload identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      tier = "regional-hub"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}


# GKE Cluster: nautilus-dr-us-west
resource "google_container_cluster" "nautilus_dr_us_west" {
  name     = "nautilus-dr-us-west"
  location = var.gcp_region
  
  # Network configuration
  network    = google_compute_network.nautilus_dr_us_west.self_link
  subnetwork = google_compute_subnetwork.nautilus_dr_us_west_private.self_link

  # Cluster configuration
  initial_node_count       = 1
  remove_default_node_pool = true
  min_master_version       = "1.28"

  # Private cluster configuration
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"

    master_global_access_config {
      enabled = true
    }
  }

  # IP allocation for VPC-native
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Network policy
  network_policy {
    enabled  = true
    provider = "CALICO"
  }

  # Workload identity
  workload_identity_config {
    workload_pool = "{var.gcp_project}.svc.id.goog"
  }

  # Logging and monitoring
  logging_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "WORKLOADS",
      "API_SERVER"
    ]
  }

  monitoring_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "WORKLOADS"
    ]
  }
}

# VPC Network
resource "google_compute_network" "nautilus_dr_us_west" {
  name                    = "nautilus-dr-us-west-vpc"
  auto_create_subnetworks = false
}

# Private Subnet  
resource "google_compute_subnetwork" "nautilus_dr_us_west_private" {
  name          = "nautilus-dr-us-west-private-subnet"
  ip_cidr_range = "10.2.0.0/16"
  region        = var.gcp_region
  network       = google_compute_network.nautilus_dr_us_west.self_link

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.12.0.0/14"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.22.0.0/16"
  }

  private_ip_google_access = true
}

# Node Pool: standby-nodes
resource "google_container_node_pool" "nautilus_dr_us_west_pool_0" {
  name       = "standby-nodes"
  cluster    = google_container_cluster.nautilus_dr_us_west.name
  location   = var.gcp_region
  node_count = 1

  autoscaling {
    min_node_count = 1
    max_node_count = 10
  }

  node_config {
    machine_type = "c2-standard-4"
    disk_size_gb = 50
    disk_type    = "pd-ssd"

    # Workload identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      tier = "disaster-recovery"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}


# GKE Cluster: nautilus-analytics-us
resource "google_container_cluster" "nautilus_analytics_us" {
  name     = "nautilus-analytics-us"
  location = var.gcp_region
  
  # Network configuration
  network    = google_compute_network.nautilus_analytics_us.self_link
  subnetwork = google_compute_subnetwork.nautilus_analytics_us_private.self_link

  # Cluster configuration
  initial_node_count       = 1
  remove_default_node_pool = true
  min_master_version       = "1.28"

  # Private cluster configuration
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"

    master_global_access_config {
      enabled = true
    }
  }

  # IP allocation for VPC-native
  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  # Network policy
  network_policy {
    enabled  = true
    provider = "CALICO"
  }

  # Workload identity
  workload_identity_config {
    workload_pool = "{var.gcp_project}.svc.id.goog"
  }

  # Logging and monitoring
  logging_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "WORKLOADS",
      "API_SERVER"
    ]
  }

  monitoring_config {
    enable_components = [
      "SYSTEM_COMPONENTS",
      "WORKLOADS"
    ]
  }
}

# VPC Network
resource "google_compute_network" "nautilus_analytics_us" {
  name                    = "nautilus-analytics-us-vpc"
  auto_create_subnetworks = false
}

# Private Subnet  
resource "google_compute_subnetwork" "nautilus_analytics_us_private" {
  name          = "nautilus-analytics-us-private-subnet"
  ip_cidr_range = "10.2.0.0/16"
  region        = var.gcp_region
  network       = google_compute_network.nautilus_analytics_us.self_link

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.12.0.0/14"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.22.0.0/16"
  }

  private_ip_google_access = true
}

# Node Pool: analytics-compute
resource "google_container_node_pool" "nautilus_analytics_us_pool_0" {
  name       = "analytics-compute"
  cluster    = google_container_cluster.nautilus_analytics_us.name
  location   = var.gcp_region
  node_count = 2

  autoscaling {
    min_node_count = 2
    max_node_count = 20
  }

  node_config {
    machine_type = "n2-highmem-8"
    disk_size_gb = 50
    disk_type    = "pd-ssd"

    # Workload identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      tier = "analytics-cluster"
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}
