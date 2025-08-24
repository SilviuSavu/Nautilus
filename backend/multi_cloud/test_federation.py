#!/usr/bin/env python3

import asyncio
import sys
from multi_cloud_federation_architect import MultiCloudFederationArchitect

async def test_federation():
    """Test the multi-cloud federation architect"""
    
    print("🌐 Testing Multi-Cloud Federation Architect")
    print("============================================")
    
    try:
        architect = MultiCloudFederationArchitect()
        
        # Design federation
        print("1. Designing federation architecture...")
        federation_config = architect.design_global_federation()
        print(f"✅ Federation designed with {len(federation_config.clusters)} clusters")
        
        # Generate manifests
        print("2. Generating Kubernetes manifests...")
        manifests = await architect.generate_kubernetes_manifests()
        print(f"✅ Generated {len(sum(manifests.values(), []))} Kubernetes manifests")
        
        # Generate Terraform configs
        print("3. Generating Terraform configurations...")
        terraform_configs = await architect.generate_terraform_infrastructure()
        print(f"✅ Generated {len(terraform_configs)} Terraform configurations")
        
        # Create DR scripts
        print("4. Creating disaster recovery scripts...")
        dr_scripts = await architect.create_disaster_recovery_automation()
        print(f"✅ Generated {len(dr_scripts)} DR scripts")
        
        # Create documentation
        print("5. Creating documentation...")
        network_docs = await architect.create_network_topology_documentation()
        print(f"✅ Generated network topology documentation ({len(network_docs)} characters)")
        
        print("\n🎉 All components generated successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_federation())
    sys.exit(0 if success else 1)