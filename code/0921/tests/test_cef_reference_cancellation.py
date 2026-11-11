"""Unbounded reference work remains explicitly interruptible in a worker pool."""
import threading
import time
import unittest


class CancellationTests(unittest.TestCase):
    def test_pool_interrupt_reaches_an_active_reference_worker(self):
        from cef.parallel import completed_tasks,cancellation_requested
        started=threading.Event();stopped=threading.Event()
        def work(item):
            if item==0:
                self.assertTrue(started.wait(2));return 'ready'
            started.set()
            deadline=time.monotonic()+2
            while not cancellation_requested() and time.monotonic()<deadline:time.sleep(.01)
            if cancellation_requested():stopped.set()
            return 'stopped'
        tasks=completed_tasks([0,1],work,2)
        self.assertEqual(next(tasks)[1],'ready')
        begin=time.monotonic()
        with self.assertRaises(KeyboardInterrupt):tasks.throw(KeyboardInterrupt())
        self.assertTrue(stopped.is_set())
        self.assertLess(time.monotonic()-begin,1.)
        self.assertFalse(cancellation_requested())

