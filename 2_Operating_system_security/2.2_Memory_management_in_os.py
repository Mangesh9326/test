#1. Simulate memory allocation
memory = [None] * 100  
def allocate(process_id, size):  
   for i in range(len(memory) - size + 1):  
       if all(block is None for block in memory[i:i+size]):  
           for j in range(i, i+size):  
               memory[j] = process_id  
       return f"Allocated {size} blocks to {process_id} at {i}"  
   return "Insufficient memory"   
def deallocate(process_id):  
   count = memory.count(process_id)  
   for i in range(len(memory)):  
       if memory[i] == process_id:  
           memory[i] = None  
       return f"Deallocated {count} blocks from {process_id}"   
print(allocate("P1", 10))  
print(allocate("P2", 20))  
print(deallocate("P1"))  



#2. Monitor real memory usage of python process
import os, psutil  
  
process = psutil.Process(os.getpid())  
print(f"Current Memory Usage: {process.memory_info().rss /  1024**2:.2f} MB") 



#3. Garbage Collection Visualization Code:  
import gc  
import sys  
  
a = [1] * (10**6)  
print("Memory used:", sys.getsizeof(a), "bytes") 
del a  
gc.collect()  
print("Garbage collected") 



#4. Paging Simulation (Basic)  
import math  
  
def paging_simulator(process_size, page_size):  
   num_pages = math.ceil(process_size / page_size)  
   page_table = {f"Page {i}": f"Frame {i}" for i in range(num_pages)}  
   return page_table  
  
table = paging_simulator(1024, 256)
for k, v in table.items():  
   print(f"{k} → {v}") 



#5. Memory Leak Simulation + Detection (for concept only) Code:  
import tracemalloc

tracemalloc.start()  
def leak_memory():  
   leaky_list = []  
   for _ in range(100000):  
       leaky_list.append("leak" * 1000)  
   return leaky_list  
   leak_memory()  
  
snapshot = tracemalloc.take_snapshot()  
top_stats = snapshot.statistics("lineno")  
print(" Memory Leak Detected at:")  
for stat in top_stats[:3]:  
   print(stat)



#6. Simulated Buffer Overflow Code:  
def safe_array_write(array, index, value):  
   if index < len(array):  
       array[index] = value        
       return " Value written"  
   else:
       return " Buffer Overflow Attempt!"  
  
arr = [0] * 5  
print(safe_array_write(arr, 2, 10))  
print(safe_array_write(arr, 10, 10)) 
