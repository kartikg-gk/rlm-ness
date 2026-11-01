"""Background jobs and parent/child mailboxes for recursive runs."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import queue
import threading
import uuid

from .session import Session


@dataclass
class Job:
    parent: str
    keep: bool
    token: threading.Event = field(default_factory=threading.Event)
    inbox: queue.Queue = field(default_factory=queue.Queue)
    outbox: list = field(default_factory=list)
    status: str = "starting"
    value: object = None
    error: Exception | None = None
    ready: threading.Event = field(default_factory=threading.Event)
    future: object = None


class Background:
    def __init__(self, max_agents=50):
        self.jobs = {}
        self.max_agents = max_agents
        self.pool = ThreadPoolExecutor(max_workers=32, thread_name_prefix="recursive-job")
        self.lock = threading.RLock()
        self.owners = {}
        self.resolve_role = None

    def bindings(self, parent, child):
        def owned(key):
            job = self.jobs[key]
            if job.parent != parent:
                raise ValueError("job does not belong to this agent")
            return job

        def spawn(task, data=None, keep=False, role=None, model=None):
            if role:
                if self.resolve_role is None:
                    raise ValueError("no role resolver configured")
                task = self.resolve_role(role) + "\n\n" + task
            with self.lock:
                if len(self.jobs) >= self.max_agents:
                    raise RuntimeError("background agent limit reached")
                key = uuid.uuid4().hex
                job = Job(parent=parent, keep=keep)
                self.jobs[key] = job
            def work():
                state = Session() if keep else None
                question = task
                try:
                    while not job.token.is_set():
                        job.status = "running"
                        job.value = child(data, question, job.token, saved=state, owner=key, model=model)
                        job.status = "idle" if keep else "done"
                        job.ready.set()
                        if not keep:
                            return
                        question = job.inbox.get()
                        if question is None:
                            return
                except Exception as error:
                    job.error = error
                    job.status = "cancelled" if job.token.is_set() else "failed"
                finally:
                    if job.token.is_set():
                        job.status = "cancelled"
                    job.ready.set()
            job.future = self.pool.submit(work)
            return key

        def result(key, timeout=None):
            job = owned(key)
            if not job.ready.wait(timeout):
                raise TimeoutError("agent still running")
            if job.status == "cancelled":
                raise RuntimeError("agent cancelled")
            if job.error:
                raise job.error
            return job.value

        def send(key, text):
            job = owned(key)
            if job.status in ("done", "failed", "cancelled"):
                raise RuntimeError("agent already finished")
            if job.status == "idle":
                job.ready.clear()
                job.status = "starting"
            job.inbox.put(text)

        def cancel(key):
            job = owned(key)
            job.token.set()
            job.inbox.put(None)
            job.status = "cancelled"
            job.ready.set()

        def messages(key):
            job = owned(key)
            with self.lock:
                items, job.outbox = job.outbox, []
            return items

        def tell(text):
            key = self.owners.get(parent)
            if key is None:
                raise RuntimeError("no background parent mailbox")
            with self.lock:
                self.jobs[key].outbox.append(text)

        return {"_spawn": spawn, "_result": result, "_status": lambda key: owned(key).status,
                "_send": send, "_cancel": cancel, "_messages": messages,
                "_children": lambda: [k for k,v in self.jobs.items() if v.parent == parent],
                "tell_parent": tell}

    def drain(self, parent):
        key = self.owners.get(parent)
        if key is None:
            return []
        items = []
        while True:
            try:
                item = self.jobs[key].inbox.get_nowait()
            except queue.Empty:
                return items
            if item is not None:
                items.append(item)

    def close(self):
        for job in list(self.jobs.values()):
            job.token.set()
            job.inbox.put(None)
            job.ready.set()
        self.pool.shutdown(wait=False, cancel_futures=True)


PRELUDE = '''
class _AgentHandle:
    def __init__(self, id): self.id = id
    async def result(self, timeout=None): return await _result(self.id, timeout)
    async def status(self): return await _status(self.id)
    async def cancel(self): return await _cancel(self.id)
    async def send(self, text): return await _send(self.id, text)
    async def messages(self): return await _messages(self.id)
async def spawn(task, data=None, keep=False, role=None, model=None):
    return _AgentHandle(await _spawn(task, data, keep, role, model))
async def children(): return [_AgentHandle(id) for id in await _children()]
'''
