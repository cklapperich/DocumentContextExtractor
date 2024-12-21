import asyncio

async def main():
    data = [x for x in range(1000)]

    async def aiter(iterable):
        for v in iterable:
            yield v

    aiterable = aiter(data)
    aiterables = [aiterable] * 1000

    values = await asyncio.gather(
        *[it.__anext__() for it in aiterables])
    
    assert values == data, f'{values} != {data}'

loop = asyncio.get_event_loop()
loop.run_until_complete(main())