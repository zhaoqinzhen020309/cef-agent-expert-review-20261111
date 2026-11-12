"""Bounded orchestration; each solver task still owns a fresh isolated process."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import threading

_task_context=threading.local()


def cancellation_requested():
    event=getattr(_task_context,'cancel',None)
    return event is not None and event.is_set()


def _run_task(work,item,cancel):
    _task_context.cancel=cancel
    try:return work(item)
    finally:del _task_context.cancel


def completed_tasks(items, work, workers):
    """Yield (item, result); consume/submit at most `workers` unfinished items."""
    if type(workers) is not int or workers < 1:
        raise ValueError('evaluation_workers must be a positive integer')
    if workers == 1:
        for item in items:
            yield item, work(item)
        return
    iterator = iter(items)
    cancel=threading.Event()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix='cef_case') as pool:
        pending = {}
        exhausted = False
        try:
            while pending or not exhausted:
                while not exhausted and len(pending) < workers:
                    try:
                        item = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    pending[pool.submit(_run_task,work,item,cancel)] = item
                if pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        item = pending.pop(future)
                        yield item, future.result()
        except BaseException:
            cancel.set()
            for future in pending:future.cancel()
            raise
