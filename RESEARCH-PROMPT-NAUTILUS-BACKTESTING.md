# NautilusTrader Backtesting Implementation Research Prompt

## Research Objective

Gather comprehensive technical documentation and implementation guidance for building a high-performance NautilusTrader backtesting system with real-time progress tracking, optimal Docker resource allocation, and advanced equity curve visualization capabilities.

## Background Context

**Current Project**: Nautilus Trading Platform with Interactive Brokers Gateway integration
- **Architecture**: Python backend (FastAPI), React frontend, NautilusTrader core
- **Database**: PostgreSQL with nanosecond precision
- **Containerization**: Docker and Docker Compose

**Identified Knowledge Gaps**:
1. Specific NautilusTrader backtest engine API documentation location
2. Exact Docker container resource limits for concurrent backtests  
3. WebSocket message format for progress updates

**Risk Factors**:
1. Learning NautilusTrader backtest engine specifics
2. Optimizing performance for large datasets
3. Complex equity curve visualization requirements

## Research Questions

### Primary Questions (Must Answer)

1. **NautilusTrader API Documentation**
   - Where is the official NautilusTrader backtest engine API documentation?
   - What are the core API endpoints, methods, and classes for backtesting?
   - What are the required parameters, data formats, and configuration options?
   - Are there complete working code examples and tutorials available?
   - What is the recommended project structure for backtest implementations?

2. **Docker Performance Optimization**
   - What are the optimal Docker container resource limits (CPU, memory, disk) for NautilusTrader backtests?
   - How many concurrent backtest instances can run efficiently on typical hardware?
   - What are the disk I/O and network considerations for large historical datasets?
   - Are there existing Docker configurations or compose files optimized for NautilusTrader?
   - What monitoring tools work best for tracking container performance during backtests?

3. **Real-time Progress Tracking**
   - What is the exact WebSocket message format for NautilusTrader backtest progress updates?
   - What event types and lifecycle states are available for tracking?
   - How should error handling and reconnection protocols be implemented?
   - What is the recommended frequency for progress updates without impacting performance?
   - Are there built-in progress tracking mechanisms or do they need custom implementation?

### Secondary Questions (Nice to Have)

4. **Performance Best Practices**
   - How do other teams handle NautilusTrader backtest performance optimization?
   - What are common bottlenecks and their solutions?
   - What data preprocessing steps improve backtest performance?

5. **Visualization Integration**
   - What visualization libraries work best with NautilusTrader equity curve data?
   - Are there existing React components for trading performance visualization?
   - What data formats are optimal for real-time chart updates?

6. **Architecture Patterns**
   - What are proven architectural patterns for concurrent backtest execution?
   - How should results be aggregated and stored for analysis?
   - What caching strategies improve repeated backtest performance?

## Research Methodology

### Information Sources (Priority Order)

1. **Official Documentation**
   - NautilusTrader official documentation and API reference
   - GitHub repository README, wiki, and examples
   - Official installation and configuration guides

2. **Community Resources**
   - NautilusTrader Discord/Slack community discussions
   - GitHub issues and pull requests related to backtesting
   - Stack Overflow and Reddit discussions

3. **Technical Implementations**
   - Open source projects using NautilusTrader for backtesting
   - Docker Hub containers and configurations
   - Performance benchmarking studies and reports

4. **Expert Knowledge**
   - NautilusTrader maintainer responses and recommendations
   - Trading platform architecture blogs and case studies
   - Performance optimization guides for Python/Docker environments

### Analysis Frameworks

- **Technical Feasibility Matrix**: Evaluate complexity vs. benefit for each implementation approach
- **Performance Benchmarking**: Compare resource requirements across different configurations
- **Risk Assessment**: Identify potential blockers and mitigation strategies
- **Implementation Timeline**: Estimate learning curve and development effort

### Data Requirements

- **Recency**: Prefer documentation and examples from the last 12 months
- **Credibility**: Prioritize official sources and verified community contributors
- **Completeness**: Seek end-to-end examples rather than partial implementations
- **Relevance**: Focus on production-ready solutions rather than proof-of-concepts

## Expected Deliverables

### Executive Summary

- **Key Findings**: Location of essential documentation and resources
- **Critical Implementation Requirements**: Resource specifications and architectural recommendations
- **Risk Assessment**: Potential blockers and recommended mitigation strategies
- **Recommended Next Steps**: Prioritized action plan for implementation

### Detailed Analysis

#### 1. API Documentation Summary
- Complete list of relevant NautilusTrader classes and methods
- Code examples for basic and advanced backtest scenarios
- Configuration options and their performance implications

#### 2. Performance Optimization Guide
- Recommended Docker container specifications
- Concurrent execution strategies and limitations
- Dataset handling and preprocessing recommendations

#### 3. Real-time Integration Specifications
- WebSocket implementation details and message schemas
- Frontend integration patterns for progress tracking
- Error handling and recovery procedures

#### 4. Implementation Roadmap
- Phase-by-phase development approach
- Resource requirements and timeline estimates
- Testing and validation strategies

### Supporting Materials

- **Resource Comparison Table**: Docker configurations and their performance characteristics
- **API Reference Quick Guide**: Essential methods and parameters for backtesting
- **Architecture Diagrams**: Recommended system design patterns
- **Code Examples**: Working implementations for common scenarios
- **Source Documentation**: Links to all referenced materials with access dates

## Success Criteria

The research will be considered successful when:

1. **Complete API Understanding**: Clear path to implement NautilusTrader backtesting with specific code examples
2. **Performance Optimization**: Concrete resource specifications and configuration recommendations
3. **Real-time Integration**: Detailed WebSocket implementation guide with message formats
4. **Risk Mitigation**: Identified potential blockers with specific solutions or workarounds
5. **Actionable Roadmap**: Clear next steps with realistic timeline and resource estimates

## Timeline and Priority

**Phase 1 (High Priority)**: Core API documentation and basic implementation examples
**Phase 2 (Medium Priority)**: Performance optimization and Docker configuration details  
**Phase 3 (Lower Priority)**: Advanced visualization integration and community best practices

**Target Completion**: Within 1-2 days to unblock development progress

## Integration with Project Documentation

**Output Location**: This research will be documented in the project repository at:
- `/docs/research/nautilus-backtesting-research.md` (detailed findings)
- `/docs/api/nautilus-backtest-api.md` (API reference summary)
- `/docker/backtesting/README.md` (Docker configuration guide)

**Update Dependencies**: 
- Update main CLAUDE.md with backtesting command references
- Add to project documentation index
- Include in developer onboarding materials