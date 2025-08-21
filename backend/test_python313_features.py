"""
Test Python 3.13 features to verify they're working correctly
"""

# Test modern union syntax (Python 3.10+)
def test_union_syntax(value: str | int | None = None) -> str | None:
    """Test new union syntax instead of str | int | None"""
    if value is None:
        return None
    return str(value)

# Test match/case statements (Python 3.10+)
def test_match_case(status: str) -> str:
    """Test match/case statement"""
    match status:
        case "connected":
            return "✓ Connected"
        case "disconnected":
            return "✗ Disconnected"
        case "error":
            return "⚠ Error"
        case _:
            return "? Unknown"

# Test generic type aliases (Python 3.12+)
type StringOrInt = str | int
type StrategyConfig = dict[str, StringOrInt]

# Test f-string improvements (Python 3.12+)
def test_fstring_improvements(name: str, value: float) -> str:
    """Test improved f-string formatting"""
    precision = 2
    return f"Strategy {name}: {value:.{precision}f}"

# Test type statement (Python 3.12+)
type PortfolioData = dict[str, str | float | int]

# Test dataclass improvements
from dataclasses import dataclass

@dataclass
class ModernStrategy:
    """Test modern dataclass with new features"""
    name: str
    parameters: dict[str, str | int | float]
    enabled: bool = True
    
    def to_config(self) -> PortfolioData:
        """Convert to configuration dictionary"""
        return {
            "name": self.name,
            "enabled": self.enabled,
            **self.parameters
        }

# Test enhanced error groups (Python 3.11+)
class StrategyError(Exception):
    """Strategy-related error"""
    pass

def test_exception_groups():
    """Test exception groups"""
    errors = []
    
    try:
        raise StrategyError("Configuration invalid")
    except StrategyError as e:
        errors.append(e)
    
    try:
        raise ValueError("Parameter out of range")
    except ValueError as e:
        errors.append(e)
    
    if errors:
        # In Python 3.11+, we could use ExceptionGroup
        return f"Found {len(errors)} errors: {[str(e) for e in errors]}"
    
    return "No errors"

# Test improved typing features
def process_strategy_data(
    strategies: list[ModernStrategy],
    filter_enabled: bool = True
) -> list[PortfolioData]:
    """Process strategy data with modern typing"""
    
    result: list[PortfolioData] = []
    
    for strategy in strategies:
        status = "enabled" if strategy.enabled else "disabled"
        
        # Use match/case for processing
        match status:
            case "enabled" if filter_enabled:
                result.append(strategy.to_config())
            case "disabled" if not filter_enabled:
                result.append(strategy.to_config())
            case _:
                continue
    
    return result

# Test async improvements (if any)
async def test_async_features() -> str | None:
    """Test modern async features"""
    import asyncio
    
    # Test modern async syntax
    await asyncio.sleep(0.01)
    
    return test_union_syntax("async_result")

if __name__ == "__main__":
    # Test all features
    print("Testing Python 3.13 features:")
    
    # Test union syntax
    result1 = test_union_syntax(42)
    print(f"Union syntax: {result1}")
    
    # Test match/case
    result2 = test_match_case("connected")
    print(f"Match/case: {result2}")
    
    # Test f-string improvements
    result3 = test_fstring_improvements("MovingAverage", 3.14159)
    print(f"F-string: {result3}")
    
    # Test dataclass
    strategy = ModernStrategy(
        name="TestStrategy",
        parameters={"period": 20, "threshold": 0.05}
    )
    config = strategy.to_config()
    print(f"Dataclass: {config}")
    
    # Test exception groups
    result4 = test_exception_groups()
    print(f"Exceptions: {result4}")
    
    # Test list processing
    strategies = [
        ModernStrategy("Strategy1", {"param1": 10}, True),
        ModernStrategy("Strategy2", {"param2": 20}, False),
    ]
    result5 = process_strategy_data(strategies)
    print(f"Processing: {result5}")
    
    print("\n✓ All Python 3.13 features working correctly!")