
terraform {
  required_version = ">= 1.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.70"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
  }
}

provider "azurerm" {
  features {}
}

variable "azure_location" {
  description = "Azure location for deployment"
  type        = string
}

variable "resource_group_name" {
  description = "Name of the Azure resource group"
  type        = string
}


# Resource Group
resource "azurerm_resource_group" "nautilus_primary_asia_northeast" {
  name     = "nautilus-primary-asia-northeast-rg"
  location = var.azure_location
}

# AKS Cluster: nautilus-primary-asia-northeast
resource "azurerm_kubernetes_cluster" "nautilus_primary_asia_northeast" {
  name                = "nautilus-primary-asia-northeast"
  location            = azurerm_resource_group.nautilus_primary_asia_northeast.location
  resource_group_name = azurerm_resource_group.nautilus_primary_asia_northeast.name
  dns_prefix          = "nautilus-primary-asia-northeast"
  kubernetes_version  = "1.28"

  default_node_pool {
    name       = "default"
    node_count = 1
    vm_size    = "Standard_D2s_v3"
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin = "azure"
    network_policy = "azure"
  }

  tags = {
    Environment = "production"
    Tier        = "primary-trading"
  }
}

# Virtual Network
resource "azurerm_virtual_network" "nautilus_primary_asia_northeast" {
  name                = "nautilus-primary-asia-northeast-vnet"
  address_space       = ["10.5.0.0/16"]
  location            = azurerm_resource_group.nautilus_primary_asia_northeast.location
  resource_group_name = azurerm_resource_group.nautilus_primary_asia_northeast.name
}

# Subnet
resource "azurerm_subnet" "nautilus_primary_asia_northeast_private" {
  name                 = "nautilus-primary-asia-northeast-private-subnet"
  resource_group_name  = azurerm_resource_group.nautilus_primary_asia_northeast.name
  virtual_network_name = azurerm_virtual_network.nautilus_primary_asia_northeast.name
  address_prefixes     = ["10.0.1.0/24"]
}


# Resource Group
resource "azurerm_resource_group" "nautilus_hub_asia_southeast" {
  name     = "nautilus-hub-asia-southeast-rg"
  location = var.azure_location
}

# AKS Cluster: nautilus-hub-asia-southeast
resource "azurerm_kubernetes_cluster" "nautilus_hub_asia_southeast" {
  name                = "nautilus-hub-asia-southeast"
  location            = azurerm_resource_group.nautilus_hub_asia_southeast.location
  resource_group_name = azurerm_resource_group.nautilus_hub_asia_southeast.name
  dns_prefix          = "nautilus-hub-asia-southeast"
  kubernetes_version  = "1.28"

  default_node_pool {
    name       = "default"
    node_count = 1
    vm_size    = "Standard_D2s_v3"
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin = "azure"
    network_policy = "azure"
  }

  tags = {
    Environment = "production"
    Tier        = "regional-hub"
  }
}

# Virtual Network
resource "azurerm_virtual_network" "nautilus_hub_asia_southeast" {
  name                = "nautilus-hub-asia-southeast-vnet"
  address_space       = ["10.5.0.0/16"]
  location            = azurerm_resource_group.nautilus_hub_asia_southeast.location
  resource_group_name = azurerm_resource_group.nautilus_hub_asia_southeast.name
}

# Subnet
resource "azurerm_subnet" "nautilus_hub_asia_southeast_private" {
  name                 = "nautilus-hub-asia-southeast-private-subnet"
  resource_group_name  = azurerm_resource_group.nautilus_hub_asia_southeast.name
  virtual_network_name = azurerm_virtual_network.nautilus_hub_asia_southeast.name
  address_prefixes     = ["10.5.1.0/24"]
}


# Resource Group
resource "azurerm_resource_group" "nautilus_dr_eu_central" {
  name     = "nautilus-dr-eu-central-rg"
  location = var.azure_location
}

# AKS Cluster: nautilus-dr-eu-central
resource "azurerm_kubernetes_cluster" "nautilus_dr_eu_central" {
  name                = "nautilus-dr-eu-central"
  location            = azurerm_resource_group.nautilus_dr_eu_central.location
  resource_group_name = azurerm_resource_group.nautilus_dr_eu_central.name
  dns_prefix          = "nautilus-dr-eu-central"
  kubernetes_version  = "1.28"

  default_node_pool {
    name       = "default"
    node_count = 1
    vm_size    = "Standard_D2s_v3"
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin = "azure"
    network_policy = "azure"
  }

  tags = {
    Environment = "production"
    Tier        = "disaster-recovery"
  }
}

# Virtual Network
resource "azurerm_virtual_network" "nautilus_dr_eu_central" {
  name                = "nautilus-dr-eu-central-vnet"
  address_space       = ["10.4.0.0/16"]
  location            = azurerm_resource_group.nautilus_dr_eu_central.location
  resource_group_name = azurerm_resource_group.nautilus_dr_eu_central.name
}

# Subnet
resource "azurerm_subnet" "nautilus_dr_eu_central_private" {
  name                 = "nautilus-dr-eu-central-private-subnet"
  resource_group_name  = azurerm_resource_group.nautilus_dr_eu_central.name
  virtual_network_name = azurerm_virtual_network.nautilus_dr_eu_central.name
  address_prefixes     = ["10.4.1.0/24"]
}
