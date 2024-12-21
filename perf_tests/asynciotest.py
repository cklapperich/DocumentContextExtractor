import asyncio
import time

async def task(num, sem):
    async with sem:  # Acquire a slot from the semaphore
        await asyncio.sleep(1)
        return (num, time.perf_counter_ns())

async def main():
    start_time = time.perf_counter_ns()
    print(f"Starting tasks at {start_time}")
    
    # Create semaphore with max 5 concurrent tasks
    sem = asyncio.Semaphore(5)
    
    tasks = [task(i, sem) for i in range(20)]
    results = await asyncio.gather(*tasks)
    
    # Sort by timestamp and print
    for num, completion_time in sorted(results, key=lambda x: x[1]):
        relative_time = (completion_time - start_time) / 1000  # ns to μs
        print(f"Task {num:3d} completed at +{relative_time:.3f}μs")
    
    print(f"\nAll tasks completed in {(time.perf_counter_ns() - start_time)/1e9:.3f} seconds")

asyncio.run(main())