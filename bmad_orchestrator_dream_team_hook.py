#!/usr/bin/env python3
"""
BMad Orchestrator Dream Team Hook
Trigger: "dream team" - Deploys ALL available specialized agents

Hook triggers when user types "dream team" to unleash maximum parallel processing power
"""

import json
import re
from typing import Dict, List, Any

class BMadOrchestratorDreamTeam:
    """
    BMad Orchestrator Dream Team - Maximum Agent Deployment
    Automatically detects task complexity and deploys optimal agent configuration
    """
    
    def __init__(self):
        self.available_agents = {
            "🔧Mike (Backend Engineer)": {
                "specialties": ["backend_systems", "api_integration", "performance_optimization", "database_design"],
                "tools": ["Python", "FastAPI", "PostgreSQL", "Redis", "Docker"],
                "strength": "System architecture and backend optimization"
            },
            "💻James (Full Stack Developer)": {
                "specialties": ["full_stack_development", "frontend_systems", "ui_ux", "react_development"],
                "tools": ["React", "TypeScript", "Python", "Docker", "WebSocket"],
                "strength": "Complete application development and user interfaces"
            },
            "🧪Quinn (Senior Developer & QA Architect)": {
                "specialties": ["testing_frameworks", "quality_assurance", "code_architecture", "performance_testing"],
                "tools": ["Testing frameworks", "Code analysis", "Performance profiling", "CI/CD"],
                "strength": "Quality assurance and architectural excellence"
            },
            "🏃Bob (Scrum Master)": {
                "specialties": ["project_management", "agile_methodologies", "team_coordination", "process_optimization"],
                "tools": ["Project management", "Documentation", "Process automation", "Analytics"],
                "strength": "Project coordination and delivery excellence"
            },
            "🔒Alex (Security & DevOps Engineer)": {
                "specialties": ["security_hardening", "devops_automation", "container_security", "monitoring"],
                "tools": ["Docker", "Security tools", "Monitoring systems", "Network security"],
                "strength": "Security enforcement and operational excellence"
            },
            "⚡Lightning (Performance Specialist)": {
                "specialties": ["performance_optimization", "hardware_acceleration", "caching", "scalability"],
                "tools": ["Performance profiling", "M4 Max optimization", "Caching systems", "Load testing"],
                "strength": "Maximum performance and scalability optimization"
            },
            "🩺Dr. DocHealth (Documentation Specialist)": {
                "specialties": ["documentation_health", "technical_writing", "user_experience", "information_architecture", "prerequisite_analysis"],
                "tools": ["Documentation analysis", "Technical writing", "Prerequisite validation", "User experience assessment"],
                "strength": "Medical-precision documentation health assessment and treatment"
            },
            "🎯Ace (Mission Critical Specialist)": {
                "specialties": ["critical_systems", "reliability_engineering", "fault_tolerance", "emergency_response"],
                "tools": ["Reliability tools", "Monitoring", "Alerting", "Disaster recovery"],
                "strength": "Mission-critical system reliability and uptime"
            },
            "🚀Rocket (Deployment & Infrastructure)": {
                "specialties": ["infrastructure_automation", "deployment_pipelines", "cloud_architecture", "scaling"],
                "tools": ["Docker", "Kubernetes", "CI/CD", "Infrastructure as Code"],
                "strength": "Rapid deployment and infrastructure scaling"
            }
        }
    
    def detect_task_complexity(self, user_message: str) -> Dict[str, Any]:
        """
        Analyze user message to determine optimal agent deployment strategy
        """
        message_lower = user_message.lower()
        
        complexity_indicators = {
            "simple": ["fix", "update", "change", "modify", "small"],
            "moderate": ["implement", "create", "develop", "build", "enhance"],
            "complex": ["migrate", "architecture", "system-wide", "optimize", "transform"],
            "critical": ["production", "emergency", "critical", "urgent", "maximum performance"]
        }
        
        domain_indicators = {
            "backend": ["engine", "api", "database", "server", "backend"],
            "frontend": ["ui", "interface", "dashboard", "frontend", "user"],
            "performance": ["performance", "optimization", "speed", "fast", "accelerate"],
            "security": ["security", "authentication", "authorization", "secure", "protect"],
            "testing": ["test", "validation", "quality", "qa", "verify"],
            "deployment": ["deploy", "production", "infrastructure", "docker", "container"]
        }
        
        # Analyze complexity
        complexity = "simple"
        for level, indicators in complexity_indicators.items():
            if any(indicator in message_lower for indicator in indicators):
                complexity = level
        
        # Analyze domain requirements
        required_domains = []
        for domain, indicators in domain_indicators.items():
            if any(indicator in message_lower for indicator in indicators):
                required_domains.append(domain)
        
        return {
            "complexity": complexity,
            "required_domains": required_domains,
            "message_length": len(user_message),
            "urgency_keywords": sum(1 for word in ["urgent", "critical", "asap", "immediately"] if word in message_lower)
        }
    
    def select_dream_team(self, task_analysis: Dict[str, Any]) -> List[str]:
        """
        Select optimal agent team based on task analysis
        """
        complexity = task_analysis["complexity"]
        domains = task_analysis["required_domains"]
        urgency = task_analysis["urgency_keywords"]
        
        # Base team selection based on complexity
        if complexity == "critical" or urgency > 0:
            # Deploy ALL agents for critical/urgent tasks
            return list(self.available_agents.keys())
        
        elif complexity == "complex":
            # Deploy 6-8 agents for complex tasks
            base_team = [
                "🔧Mike (Backend Engineer)",
                "💻James (Full Stack Developer)", 
                "🧪Quinn (Senior Developer & QA Architect)",
                "🏃Bob (Scrum Master)",
                "🔒Alex (Security & DevOps Engineer)",
                "⚡Lightning (Performance Specialist)"
            ]
            
            # Add domain specialists
            if "deployment" in domains or "infrastructure" in domains:
                base_team.append("🚀Rocket (Deployment & Infrastructure)")
            if urgency > 0 or "critical" in domains:
                base_team.append("🎯Ace (Mission Critical Specialist)")
            
            return base_team
        
        elif complexity == "moderate":
            # Deploy 4-6 agents for moderate tasks
            base_team = [
                "🔧Mike (Backend Engineer)",
                "💻James (Full Stack Developer)",
                "🧪Quinn (Senior Developer & QA Architect)",
                "🏃Bob (Scrum Master)"
            ]
            
            # Add specialists based on domain
            if "performance" in domains:
                base_team.append("⚡Lightning (Performance Specialist)")
            if "security" in domains:
                base_team.append("🔒Alex (Security & DevOps Engineer)")
            
            return base_team
        
        else:
            # Simple tasks - deploy 2-3 core agents
            return [
                "🔧Mike (Backend Engineer)",
                "💻James (Full Stack Developer)",
                "🧪Quinn (Senior Developer & QA Architect)"
            ]
    
    def generate_agent_deployment(self, selected_agents: List[str], user_message: str) -> str:
        """
        Generate the agent deployment commands and briefing
        """
        deployment_script = f"""
## 🚀 BMad Orchestrator - Dream Team Deployment Activated!

**User Request**: {user_message}

**Dream Team Composition**: {len(selected_agents)} specialized agents deployed

### 🎯 Agent Deployment Strategy

"""
        
        for i, agent in enumerate(selected_agents, 1):
            agent_info = self.available_agents[agent]
            deployment_script += f"""
#### Agent {i}: {agent}
- **Specialty**: {agent_info['strength']}
- **Core Skills**: {', '.join(agent_info['specialties'])}
- **Tools**: {', '.join(agent_info['tools'])}

**Mission Briefing**: {self._generate_mission_briefing(agent, user_message)}

```bash
export BMAD_AGENT="{agent}"
```

**Agent Task Assignment**:
```
You are {agent} - Specialized agent deployed by BMad Orchestrator Dream Team!

CRITICAL MISSION: {user_message}

Your Specific Role:
{self._generate_role_specific_instructions(agent, user_message)}

Success Criteria:
- Complete your specialized portion of the mission
- Coordinate with other Dream Team agents
- Report detailed progress and results
- Ensure maximum performance and quality

Dream Team Collaboration: You are working alongside {len(selected_agents)-1} other specialized agents. Focus on your expertise while supporting the overall mission success.

Return comprehensive report of your contribution to the Dream Team mission.
```

"""
        
        deployment_script += f"""
### 🎉 Dream Team Deployment Complete!

**Total Agents**: {len(selected_agents)} specialized agents
**Deployment Strategy**: Parallel execution with specialized coordination
**Expected Outcome**: Maximum efficiency through expert collaboration

**BMad Orchestrator Status**: Dream Team activated - All agents deploying for mission success!
"""
        
        return deployment_script
    
    def _generate_mission_briefing(self, agent: str, user_message: str) -> str:
        """Generate specific mission briefing for each agent"""
        briefings = {
            "🔧Mike (Backend Engineer)": "Handle all backend systems, APIs, and performance optimization aspects",
            "💻James (Full Stack Developer)": "Manage full-stack development, UI/UX, and user-facing components", 
            "🧪Quinn (Senior Developer & QA Architect)": "Ensure quality, testing, and architectural excellence",
            "🏃Bob (Scrum Master)": "Coordinate project delivery and optimize team processes",
            "🔒Alex (Security & DevOps Engineer)": "Implement security measures and deployment automation",
            "⚡Lightning (Performance Specialist)": "Maximize system performance and scalability",
            "🎯Ace (Mission Critical Specialist)": "Ensure reliability and handle critical system requirements",
            "🚀Rocket (Deployment & Infrastructure)": "Manage deployment pipelines and infrastructure scaling"
        }
        return briefings.get(agent, "Contribute your specialized expertise to the mission")
    
    def _generate_role_specific_instructions(self, agent: str, user_message: str) -> str:
        """Generate role-specific instructions for each agent"""
        instructions = {
            "🔧Mike (Backend Engineer)": """
- Analyze backend architecture requirements
- Optimize database and API performance  
- Implement backend security measures
- Ensure scalable system design
- Focus on technical implementation excellence""",
            
            "💻James (Full Stack Developer)": """
- Design and implement user interfaces
- Ensure frontend-backend integration
- Optimize user experience and performance
- Handle full-stack development tasks
- Focus on complete application functionality""",
            
            "🧪Quinn (Senior Developer & QA Architect)": """
- Design comprehensive testing strategies
- Ensure code quality and best practices
- Validate system architecture decisions
- Implement quality assurance processes
- Focus on reliability and maintainability""",
            
            "🏃Bob (Scrum Master)": """
- Coordinate Dream Team activities
- Optimize project workflow and delivery
- Ensure clear communication and documentation
- Manage project timeline and deliverables
- Focus on successful mission completion""",
            
            "🔒Alex (Security & DevOps Engineer)": """
- Implement security hardening measures
- Automate deployment and operational processes
- Design monitoring and alerting systems
- Ensure compliance and security standards
- Focus on operational excellence and security""",
            
            "⚡Lightning (Performance Specialist)": """
- Identify and optimize performance bottlenecks
- Implement caching and acceleration strategies
- Design scalable high-performance solutions
- Benchmark and validate performance improvements
- Focus on maximum system performance""",
            
            "🎯Ace (Mission Critical Specialist)": """
- Ensure mission-critical system reliability
- Implement fault tolerance and recovery systems
- Design emergency response procedures
- Validate system resilience and uptime
- Focus on zero-downtime operations""",
            
            "🚀Rocket (Deployment & Infrastructure)": """
- Design automated deployment pipelines
- Implement infrastructure as code
- Optimize container and orchestration strategies
- Ensure scalable infrastructure deployment
- Focus on rapid and reliable deployment"""
        }
        return instructions.get(agent, "Apply your specialized expertise to achieve mission success")

def trigger_dream_team(user_message: str) -> str:
    """
    Main hook function - triggered when user types "dream team"
    """
    orchestrator = BMadOrchestratorDreamTeam()
    
    # Analyze the task
    task_analysis = orchestrator.detect_task_complexity(user_message)
    
    # Select optimal dream team
    selected_agents = orchestrator.select_dream_team(task_analysis)
    
    # Generate deployment
    deployment = orchestrator.generate_agent_deployment(selected_agents, user_message)
    
    return deployment

# Hook configuration for Claude Code
def dream_team_hook(message: str) -> str:
    """
    Claude Code hook function
    Triggers when user message contains "dream team"
    """
    if "dream team" in message.lower():
        return trigger_dream_team(message)
    return ""

# Example usage and testing
if __name__ == "__main__":
    # Test examples
    test_messages = [
        "dream team: optimize the entire trading system for maximum performance",
        "dream team: migrate all engines to new architecture", 
        "dream team: deploy critical security updates to production",
        "dream team: create comprehensive monitoring dashboard"
    ]
    
    for msg in test_messages:
        print("="*80)
        print(f"INPUT: {msg}")
        print("="*80)
        print(trigger_dream_team(msg))
        print("\n\n")