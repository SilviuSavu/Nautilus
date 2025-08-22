# Nautilus Trader Version Check
# Add this to your crontab for monthly monitoring
0 9 1 * * /usr/bin/curl -s https://api.github.com/repos/nautechsystems/nautilus_trader/releases/latest | grep 'tag_name' | cut -d'"' -f4

# Quick compatibility test command
docker exec nautilus-backend python -c "
try:
    from nautilus_trader.adapters.interactive_brokers.config import InteractiveBrokersDataClientConfig
    print('✅ Nautilus IB adapter compatible')
except ImportError as e:
    print(f'❌ Compatibility issue: {e}')
"
