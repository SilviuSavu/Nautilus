# Source Tree Structure

## Root Level
```
/
├── backend/           # Python backend services
├── frontend/          # React/TypeScript frontend
├── nautilus_trader/   # Core Nautilus trading platform
├── docker-compose.yml # Container orchestration
└── docs/             # Project documentation
```

## Backend (`/backend/`)
- `main.py` - Main application entry point
- `messagebus_client.py` - MessageBus integration
- `tests/` - Backend test suite

## Frontend (`/frontend/`)
- `src/` - TypeScript source code
  - `components/` - React components
  - `services/` - API and service layers
  - `types/` - TypeScript type definitions
- `public/` - Static assets and HTML files
- `tests/` - Frontend test suite

## Nautilus Trader (`/nautilus_trader/`)
- Core trading platform with extensive Rust/Python integration
- `crates/` - Rust crates for high-performance components
- `nautilus_trader/` - Python package
- `docs/` - Platform-specific documentation

## Key Patterns
- Microservices architecture with Docker containers
- Event-driven communication via MessageBus
- Rust for performance-critical components
- Python for business logic and integrations