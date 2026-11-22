import os

import psutil
import pytest

from server.runtime import host_memory
from server.runtime.runtime_pool import RuntimePool


def test_process_memory_reports_own_resident_pages():
    assert RuntimePool._process_resident_bytes(os.getpid()) > 0


@pytest.mark.parametrize("pid", [0, -1, True, None])
def test_invalid_process_identity_is_not_queried(monkeypatch, pid):
    def query(_pid):
        pytest.fail("invalid PID reached the operating system")

    monkeypatch.setattr(host_memory.psutil, "Process", query)
    assert host_memory.process_resident_bytes(pid) is None


@pytest.mark.parametrize("error", [psutil.NoSuchProcess(123), psutil.AccessDenied(123), OSError()])
def test_unavailable_process_memory_remains_unknown(monkeypatch, error):
    def query(_pid):
        raise error

    monkeypatch.setattr(host_memory.psutil, "Process", query)
    assert host_memory.process_resident_bytes(123) is None
