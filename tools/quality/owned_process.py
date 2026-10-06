"""Hidden, owned measurement child trees; no external process termination."""
import ctypes as C
from ctypes import wintypes as W
import os
from pathlib import Path
import subprocess
import time


class OwnedTreeFatalError(RuntimeError):
    """Ownership setup or confirmed tree drain failed; no subsequent dispatch is safe."""
    def __init__(self, message, *, child_pid=None, job_handle=None):
        super().__init__(message)
        self.child_pid = child_pid
        self.job_handle = job_handle


# Retain failed JOB handles while the owning Python process remains alive.
UNRESOLVED_JOB_HANDLES = {}


def process_alive(pid):
    if os.name != 'nt':
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
    kernel = C.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
    kernel.OpenProcess.restype = W.HANDLE
    kernel.GetExitCodeProcess.argtypes = [W.HANDLE, C.POINTER(W.DWORD)]
    kernel.CloseHandle.argtypes = [W.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return C.get_last_error() != 87
    try:
        code = W.DWORD()
        if not kernel.GetExitCodeProcess(handle, C.byref(code)):
            return True
        return code.value == 259
    finally:
        kernel.CloseHandle(handle)


def run_owned(command, *, environment=None, stdout, stderr, timeout, cwd=None):
    """Assign suspended Windows child to kill-on-close JOB before execution."""
    with Path(stdout).open('xb') as out, Path(stderr).open('xb') as err:
        if os.name != 'nt':
            process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=out, stderr=err,
                                       start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except BaseException:
                import signal
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30)
                except BaseException as error:
                    raise OwnedTreeFatalError('owned process group termination/drain failed', child_pid=process.pid) from error
                raise
            if code:
                raise subprocess.CalledProcessError(code, command)
            return
        kernel = C.WinDLL('kernel32', use_last_error=True)
        size = C.c_size_t
        class Limits(C.Structure):
            _fields_ = [('process_time', C.c_longlong), ('job_time', C.c_longlong),
                        ('flags', W.DWORD), ('min_ws', size), ('max_ws', size),
                        ('active_limit', W.DWORD), ('affinity', size), ('priority', W.DWORD), ('scheduling', W.DWORD)]
        class IO(C.Structure):
            _fields_ = [(name, C.c_ulonglong) for name in ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]
        class Extended(C.Structure):
            _fields_ = [('basic', Limits), ('io', IO), ('process_memory', size), ('job_memory', size), ('peak_process', size), ('peak_job', size)]
        class Accounting(C.Structure):
            _fields_ = [('user', C.c_longlong), ('kernel', C.c_longlong), ('period_user', C.c_longlong),
                        ('period_kernel', C.c_longlong), ('faults', W.DWORD), ('processes', W.DWORD),
                        ('active', W.DWORD), ('terminated', W.DWORD)]
        class Thread(C.Structure):
            _fields_ = [('size', W.DWORD), ('usage', W.DWORD), ('id', W.DWORD), ('owner', W.DWORD), ('base', W.LONG), ('delta', W.LONG), ('flags', W.DWORD)]
        kernel.CreateJobObjectW.argtypes = [C.c_void_p, W.LPCWSTR]
        kernel.CreateJobObjectW.restype = W.HANDLE
        kernel.SetInformationJobObject.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
        kernel.TerminateJobObject.argtypes = [W.HANDLE, W.UINT]
        kernel.QueryInformationJobObject.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.c_void_p]
        kernel.CreateToolhelp32Snapshot.argtypes = [W.DWORD, W.DWORD]
        kernel.CreateToolhelp32Snapshot.restype = W.HANDLE
        kernel.Thread32First.argtypes = [W.HANDLE, C.POINTER(Thread)]
        kernel.Thread32Next.argtypes = [W.HANDLE, C.POINTER(Thread)]
        kernel.OpenThread.argtypes = [W.DWORD, W.BOOL, W.DWORD]
        kernel.OpenThread.restype = W.HANDLE
        kernel.ResumeThread.argtypes = [W.HANDLE]
        kernel.ResumeThread.restype = W.DWORD
        kernel.CloseHandle.argtypes = [W.HANDLE]
        def check(value):
            if not value:
                raise C.WinError(C.get_last_error())
            return value
        def ownership_check(value):
            try:
                return check(value)
            except OSError as error:
                raise OwnedTreeFatalError('owned JOB setup failed: ' + str(error)) from error
        job = ownership_check(kernel.CreateJobObjectW(None, None))
        child, assigned = None, False
        try:
            limits = Extended()
            limits.basic.flags = 0x2000
            ownership_check(kernel.SetInformationJobObject(job, 9, C.byref(limits), C.sizeof(limits)))
            child = subprocess.Popen(command, env=environment, cwd=cwd, stdout=out, stderr=err,
                                     creationflags=subprocess.CREATE_NO_WINDOW | 4)
            ownership_check(kernel.AssignProcessToJobObject(job, W.HANDLE(int(child._handle))))
            assigned = True
            snapshot = kernel.CreateToolhelp32Snapshot(4, 0)
            if snapshot == C.c_void_p(-1).value:
                raise OwnedTreeFatalError('owned suspended thread inspection failed', child_pid=child.pid, job_handle=int(job))
            try:
                thread = Thread()
                thread.size = C.sizeof(thread)
                found = kernel.Thread32First(snapshot, C.byref(thread))
                resumed = False
                while found:
                    if thread.owner == child.pid:
                        handle = ownership_check(kernel.OpenThread(2, False, thread.id))
                        try:
                            if kernel.ResumeThread(handle) == 0xffffffff:
                                raise OwnedTreeFatalError('owned suspended thread resume failed', child_pid=child.pid, job_handle=int(job))
                            resumed = True
                            break
                        finally:
                            kernel.CloseHandle(handle)
                    found = kernel.Thread32Next(snapshot, C.byref(thread))
                if not resumed:
                    raise OwnedTreeFatalError('owned suspended primary thread missing', child_pid=child.pid, job_handle=int(job))
            finally:
                kernel.CloseHandle(snapshot)
            code = child.wait(timeout=timeout)
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            try:
                if child is not None and not assigned:
                    child.kill()
                check(kernel.TerminateJobObject(job, 1))
                deadline = time.monotonic() + 30
                while True:
                    state = Accounting()
                    check(kernel.QueryInformationJobObject(job, 1, C.byref(state), C.sizeof(state), None))
                    if not state.active:
                        break
                    if time.monotonic() > deadline:
                        raise TimeoutError('owned JOB failed to drain')
                    time.sleep(0.02)
                if child is not None:
                    child.wait(timeout=30)
                check(kernel.CloseHandle(job))
            except BaseException as error:
                pid = child.pid if child is not None else None
                UNRESOLVED_JOB_HANDLES[int(job)] = pid
                raise OwnedTreeFatalError('owned JOB termination/drain failed; retain handle and stop measurement: ' + str(error),
                                          child_pid=pid, job_handle=int(job)) from error
