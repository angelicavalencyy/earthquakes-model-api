import asyncio
import asyncpg

async def test():
    try:
        conn = await asyncpg.connect(user='fastapi', password='Angel141003', database='fastapi', host='127.0.0.1', port=5432)
        print('CONNECTED')
        await conn.close()
    except Exception as e:
        print('ERROR', repr(e))

if __name__ == '__main__':
    asyncio.run(test())
