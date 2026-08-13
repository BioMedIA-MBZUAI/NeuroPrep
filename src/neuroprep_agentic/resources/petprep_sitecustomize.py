"""Make Python multiprocessing temp cleanup safe on task-scoped NFS storage.

PETPrep uses multiprocessing managers.  At interpreter shutdown, an NFS client
may still hold a temporary socket through a ``.nfs*`` placeholder.  Python's
default cleanup raises EBUSY/ENOTEMPTY in that situation and changes an otherwise
successful PETPrep run into exit code 1.  Ignore only those two NFS cleanup
conditions; all computation and artifact errors continue to fail normally.
"""

from __future__ import annotations

import errno
from multiprocessing import process, util


def _nfs_safe_remove_temp_dir(rmtree, tempdir):
    def onerror(_func, _path, error_info):
        error = error_info[1]
        if isinstance(error, FileNotFoundError):
            return
        if isinstance(error, OSError) and error.errno in {errno.EBUSY, errno.ENOTEMPTY}:
            return
        raise error

    rmtree(tempdir, onerror=onerror)
    current_process = process.current_process()
    if current_process is not None:
        current_process._config["tempdir"] = None


util._remove_temp_dir = _nfs_safe_remove_temp_dir
