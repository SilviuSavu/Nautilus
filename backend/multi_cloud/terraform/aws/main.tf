
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes" 
      version = "~> 2.20"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
}


# EKS Cluster: nautilus-primary-us-east
resource "aws_eks_cluster" "nautilus_primary_us_east" {
  name     = "nautilus-primary-us-east"
  role_arn = aws_iam_role.nautilus_primary_us_east_cluster.arn
  version  = "1.28"

  vpc_config {
    subnet_ids              = aws_subnet.nautilus_primary_us_east_private[*].id
    endpoint_private_access = true
    endpoint_public_access  = false
    public_access_cidrs    = ["0.0.0.0/0"]
  }

  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  depends_on = [
    aws_iam_role_policy_attachment.nautilus_primary_us_east_cluster_policy,
    aws_iam_role_policy_attachment.nautilus_primary_us_east_vpc_resource_controller,
  ]

  tags = {
    Name = "nautilus-primary-us-east"
    Tier = "primary-trading"
    Federation = "nautilus-global"
  }
}

# VPC for nautilus-primary-us-east
resource "aws_vpc" "nautilus_primary_us_east" {
  cidr_block           = "10.1.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "nautilus-primary-us-east-vpc"
    "kubernetes.io/cluster/nautilus-primary-us-east" = "shared"
  }
}

# Private Subnets
resource "aws_subnet" "nautilus_primary_us_east_private" {
  count = 3
  vpc_id            = aws_vpc.nautilus_primary_us_east.id
  cidr_block        = "10.1.{count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "nautilus-primary-us-east-private-subnet-{count.index + 1}"
    "kubernetes.io/cluster/nautilus-primary-us-east" = "owned"
    "kubernetes.io/role/internal-elb" = "1"
  }
}

# Node Groups

resource "aws_eks_node_group" "nautilus_primary_us_east_node_group_0" {
  cluster_name    = aws_eks_cluster.nautilus_primary_us_east.name
  node_group_name = "ultra-low-latency"
  node_role_arn   = aws_iam_role.nautilus_primary_us_east_node.arn
  subnet_ids      = aws_subnet.nautilus_primary_us_east_private[*].id
  instance_types  = ["c6i.4xlarge"]

  scaling_config {
    desired_size = 5
    max_size     = 15
    min_size     = 5
  }

  disk_size = 200

  tags = {
    Name = "nautilus-primary-us-east-ultra-low-latency"
  }
}


# EKS Cluster: nautilus-hub-us-west
resource "aws_eks_cluster" "nautilus_hub_us_west" {
  name     = "nautilus-hub-us-west"
  role_arn = aws_iam_role.nautilus_hub_us_west_cluster.arn
  version  = "1.28"

  vpc_config {
    subnet_ids              = aws_subnet.nautilus_hub_us_west_private[*].id
    endpoint_private_access = true
    endpoint_public_access  = false
    public_access_cidrs    = ["0.0.0.0/0"]
  }

  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  depends_on = [
    aws_iam_role_policy_attachment.nautilus_hub_us_west_cluster_policy,
    aws_iam_role_policy_attachment.nautilus_hub_us_west_vpc_resource_controller,
  ]

  tags = {
    Name = "nautilus-hub-us-west"
    Tier = "regional-hub"
    Federation = "nautilus-global"
  }
}

# VPC for nautilus-hub-us-west
resource "aws_vpc" "nautilus_hub_us_west" {
  cidr_block           = "10.2.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "nautilus-hub-us-west-vpc"
    "kubernetes.io/cluster/nautilus-hub-us-west" = "shared"
  }
}

# Private Subnets
resource "aws_subnet" "nautilus_hub_us_west_private" {
  count = 3
  vpc_id            = aws_vpc.nautilus_hub_us_west.id
  cidr_block        = "10.2.{count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "nautilus-hub-us-west-private-subnet-{count.index + 1}"
    "kubernetes.io/cluster/nautilus-hub-us-west" = "owned"
    "kubernetes.io/role/internal-elb" = "1"
  }
}

# Node Groups

resource "aws_eks_node_group" "nautilus_hub_us_west_node_group_0" {
  cluster_name    = aws_eks_cluster.nautilus_hub_us_west.name
  node_group_name = "high-performance"
  node_role_arn   = aws_iam_role.nautilus_hub_us_west_node.arn
  subnet_ids      = aws_subnet.nautilus_hub_us_west_private[*].id
  instance_types  = ["c6i.2xlarge"]

  scaling_config {
    desired_size = 3
    max_size     = 8
    min_size     = 3
  }

  disk_size = 50

  tags = {
    Name = "nautilus-hub-us-west-high-performance"
  }
}


# EKS Cluster: nautilus-dr-asia-australia
resource "aws_eks_cluster" "nautilus_dr_asia_australia" {
  name     = "nautilus-dr-asia-australia"
  role_arn = aws_iam_role.nautilus_dr_asia_australia_cluster.arn
  version  = "1.28"

  vpc_config {
    subnet_ids              = aws_subnet.nautilus_dr_asia_australia_private[*].id
    endpoint_private_access = true
    endpoint_public_access  = false
    public_access_cidrs    = ["0.0.0.0/0"]
  }

  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  depends_on = [
    aws_iam_role_policy_attachment.nautilus_dr_asia_australia_cluster_policy,
    aws_iam_role_policy_attachment.nautilus_dr_asia_australia_vpc_resource_controller,
  ]

  tags = {
    Name = "nautilus-dr-asia-australia"
    Tier = "disaster-recovery"
    Federation = "nautilus-global"
  }
}

# VPC for nautilus-dr-asia-australia
resource "aws_vpc" "nautilus_dr_asia_australia" {
  cidr_block           = "10.10.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "nautilus-dr-asia-australia-vpc"
    "kubernetes.io/cluster/nautilus-dr-asia-australia" = "shared"
  }
}

# Private Subnets
resource "aws_subnet" "nautilus_dr_asia_australia_private" {
  count = 3
  vpc_id            = aws_vpc.nautilus_dr_asia_australia.id
  cidr_block        = "10.10.{count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name = "nautilus-dr-asia-australia-private-subnet-{count.index + 1}"
    "kubernetes.io/cluster/nautilus-dr-asia-australia" = "owned"
    "kubernetes.io/role/internal-elb" = "1"
  }
}

# Node Groups

resource "aws_eks_node_group" "nautilus_dr_asia_australia_node_group_0" {
  cluster_name    = aws_eks_cluster.nautilus_dr_asia_australia.name
  node_group_name = "standby-nodes"
  node_role_arn   = aws_iam_role.nautilus_dr_asia_australia_node.arn
  subnet_ids      = aws_subnet.nautilus_dr_asia_australia_private[*].id
  instance_types  = ["c6i.large"]

  scaling_config {
    desired_size = 1
    max_size     = 6
    min_size     = 1
  }

  disk_size = 50

  tags = {
    Name = "nautilus-dr-asia-australia-standby-nodes"
  }
}


# IAM Role for EKS Cluster
resource "aws_iam_role" "cluster" {
  name = "nautilus-eks-cluster-role"

  assume_role_policy = jsonencode({
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "eks.amazonaws.com"
      }
    }]
    Version = "2012-10-17"
  })
}

# IAM Role Attachments
resource "aws_iam_role_policy_attachment" "cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.cluster.name
}

resource "aws_iam_role_policy_attachment" "vpc_resource_controller" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSVPCResourceController"
  role       = aws_iam_role.cluster.name
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}
