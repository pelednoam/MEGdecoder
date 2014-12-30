'''
Some convenince methods for use with multiprocessing.Pool.
The code is taken from here:
https://github.com/dougalsutherland/py-sdm/blob/master/sdm/mp_utils.py
http://stackoverflow.com/questions/15118344/system-error-while-running-subprocesses-using-multiprocessing
http://www.pressinganswer.com/1754529/system-error-while-running-subprocesses-using-multiprocessing
'''
from __future__ import division, print_function

from contextlib import contextmanager
import itertools
import multiprocessing as mp
import os
import random
import string
from itertools import imap,izip
# from .utils import strict_map, imap, izip


def _apply(func_args):
    func, args = func_args
    return func(*args)


### Dummy implementation of (some of) multiprocessing.Pool that doesn't even
### thread (unlike multiprocessing.dummy).
class ImmediateResult(object):
    "Duck-type like multiprocessing.pool.MapResult."
    def __init__(self, value):
        self.value = value

    def get(self, timeout=None):
        return self.value

    def wait(self, timeout=None):
        pass

    def ready(self):
        return True

    def successful(self):
        return True


class DummyPool(object):
    "Duck-type like multiprocessing.Pool, mostly."
    def close(self):
        pass

    def join(self):
        pass

    def apply_async(self, func, args, kwds=None, callback=None):
        val = func(*args, **(kwds or {}))
        callback(val)
        return ImmediateResult(val)

    def map(self, func, args, chunksize=None):
        return map(func, args)

    def imap(self, func, args, chunksize=None):
        return imap(func, args)

    def imap_unordered(self, func, args, chunksize=None):
        return imap(func, args)


def patch_starmap(pool):
    '''
    A function that adds the equivalent of multiprocessing.Pool.starmap
    to a given pool if it doesn't have the function.
    '''
    if hasattr(pool, 'starmap'):
        return

    def starmap(func, iterables):
        return pool.map(_apply, izip(itertools.repeat(func), iterables))
    pool.starmap = starmap


def make_pool(n_proc=None):
    "Makes a multiprocessing.Pool or a DummyPool depending on n_proc."
    pool = DummyPool() if n_proc == 1 else mp.Pool(n_proc)
    patch_starmap(pool)
    return pool


@contextmanager
def get_pool(n_proc=None):
    "A context manager that opens a pool and joins it on exit."
    pool = make_pool(n_proc)
    yield pool
    pool.close()
    pool.join()


### A helper for letting the forked processes use data without pickling.
_data_name_cands = (
    '_data_' + ''.join(random.sample(string.ascii_lowercase, 10))
    for _ in itertools.count())


class ForkedData(object):
    '''
    Class used to pass data to child processes in multiprocessing without
    really pickling/unpickling it. Only works on POSIX.

    Intended use:
        - The master process makes the data somehow, and does e.g.
            data = ForkedData(the_value)
        - The master makes sure to keep a reference to the ForkedData object
          until the children are all done with it, since the global reference
          is deleted to avoid memory leaks when the ForkedData object dies.
        - Master process constructs a multiprocessing.Pool *after*
          the ForkedData construction, so that the forked processes
          inherit the new global.
        - Master calls e.g. pool.map with data as an argument.
        - Child gets the real value through data.value, and uses it read-only.
          Modifying it won't crash, but changes won't be propagated back to the
          master or to other processes, since it's copy-on-write.
    '''
    # TODO: more flexible garbage collection options
    def __init__(self, val):
        g = globals()
        self.name = next(n for n in _data_name_cands if n not in g)
        g[self.name] = val
        self.master_pid = os.getpid()

    def __getstate__(self):
        if os.name != 'posix':
            raise RuntimeError("ForkedData only works on OSes with fork()")
        return self.__dict__

    @property
    def value(self):
        return globals()[self.name]

    def __del__(self):
        if os.getpid() == self.master_pid:
            del globals()[self.name]


