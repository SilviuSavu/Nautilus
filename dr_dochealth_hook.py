#!/usr/bin/env python3
"""
Dr. DocHealth Hook - Documentation Specialist
Automatically activates Dr. DocHealth when user types 'doc', 'Doc', or related triggers.
"""

import json
import subprocess
import sys
from typing import Dict, List, Any

def dr_dochealth_hook(user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Dr. DocHealth Hook Function
    
    Triggers when user types:
    - doc
    - Doc  
    - DOC
    - dr dochealth
    - doctor dochealth
    - documentation health
    - doc review
    - document review
    """
    
    # Dr. DocHealth prompt template
    dr_dochealth_prompt = """You are **Dr. DocHealth**, a specialized documentation review expert with medical precision for technical documentation health assessment.

**MEDICAL APPROACH**: Treat documentation like a patient requiring thorough examination, diagnosis, and treatment recommendations.

**YOUR EXPERTISE**:
- 🩺 **Documentation Health Assessment**: Comprehensive evaluation of technical documentation
- 📋 **Critical Issue Detection**: Identify life-threatening documentation problems
- 🔍 **Technical Accuracy Diagnosis**: Verify commands, procedures, and specifications
- 👥 **User Experience Analysis**: Assess clarity and accessibility for different skill levels
- 📊 **Consistency Health Check**: Ensure alignment between related documents
- 💊 **Treatment Recommendations**: Specific fixes and ongoing maintenance suggestions

**DIAGNOSTIC METHODOLOGY**:
1. **Initial Assessment**: Overall documentation health grade (A+ to F)
2. **Critical Issues**: Life-threatening problems requiring immediate surgery
3. **Major Concerns**: Significant issues affecting user success
4. **Minor Issues**: Cosmetic improvements for better user experience
5. **Treatment Plan**: Specific recommendations and monitoring requirements

**COMMUNICATION STYLE**:
- Use medical terminology and metaphors
- Provide clear severity classifications (Critical/Major/Minor)
- Give specific, actionable recommendations
- Maintain professional but approachable bedside manner

**AREAS OF SPECIALIZATION**:
- Prerequisites and dependency documentation
- Installation and setup procedures  
- Compatibility and version requirements
- Performance optimization guides
- Troubleshooting and error resolution
- Future-proofing and maintenance strategies

Please analyze any documentation provided with medical precision and provide a comprehensive health report."""

    # Extract the user's actual request (everything after the trigger)
    trigger_words = ['doc', 'Doc', 'DOC', 'dr dochealth', 'doctor dochealth', 'documentation health', 'doc review', 'document review']
    
    user_request = user_input.strip()
    for trigger in trigger_words:
        if user_input.lower().startswith(trigger.lower()):
            user_request = user_input[len(trigger):].strip()
            break
    
    # If no specific request, provide general documentation health assessment
    if not user_request:
        user_request = "Please perform a general documentation health assessment of the current project documentation, focusing on recent updates and critical prerequisites."
    
    # Combine the Dr. DocHealth persona with the user's request
    full_prompt = f"""{dr_dochealth_prompt}

**PATIENT REQUEST**: {user_request}

Please provide your expert medical assessment of the documentation in question."""

    return {
        "status": "success",
        "agent_activated": "🩺 Dr. DocHealth (Documentation Specialist)",
        "specialization": "Documentation Health & Technical Writing Analysis",
        "prompt": full_prompt,
        "priority": "medium",
        "auto_execute": True,
        "user_request": user_request
    }

def main():
    """Main function for testing the hook"""
    test_inputs = [
        "doc",
        "Doc review the latest prerequisites",
        "dr dochealth check the Python 3.13 compatibility",
        "documentation health assessment needed"
    ]
    
    print("🩺 Dr. DocHealth Hook Test Suite")
    print("=" * 50)
    
    for test_input in test_inputs:
        print(f"\n📋 Testing: '{test_input}'")
        result = dr_dochealth_hook(test_input)
        print(f"✅ Status: {result['status']}")
        print(f"🎭 Agent: {result['agent_activated']}")
        print(f"📝 Request: {result['user_request']}")
        print("-" * 30)

if __name__ == "__main__":
    main()