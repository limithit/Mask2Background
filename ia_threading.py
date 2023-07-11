import threading
import gc
from modules.devices import torch_gc
from functools import wraps

backup_ckpt_info = None
model_access_sem = threading.Semaphore(1)

def clear_cache():
    gc.collect()
    torch_gc()





def clear_cache_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        clear_cache()
        res = func(*args, **kwargs)
        clear_cache()
        return res
    return wrapper
