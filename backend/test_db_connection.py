#!/usr/bin/env python3
"""Test database connection"""
import asyncio
import asyncpg

async def test_db():
    try:
        conn = await asyncpg.connect('postgresql://nautilus:nautilus123@localhost:5432/nautilus')
        result = await conn.fetchrow('SELECT 1 as test')
        print(f'✅ Database connection successful: {result}')
        await conn.close()
        return True
    except Exception as e:
        print(f'❌ Database connection failed: {e}')
        return False

if __name__ == "__main__":
    asyncio.run(test_db())