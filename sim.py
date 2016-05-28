import math
import random
from collections import defaultdict


class Scheduler:
    def __init__(self, sample):
        self.time = 0
        self.timers = defaultdict(list)
        self._sample = sample

    def sleep(self, duration, cb, *a, **k):
        self.timers[self.time + duration].append((cb, a, k))

    def run_for(self, duration):
        while self.time < duration:
            next_event = sorted(self.timers.keys())[0]
            assert next_event >= self.time
            self._sample(self.time, next_event)
            self.time = next_event
            for cb, a, k in self.timers.pop(next_event):
                cb(*a, **k)


class Worker:
    def __init__(self, get_job, time_to_next, max_jobs=20):
        self._get_job = get_job
        self._ttn = time_to_next
        self._max = max_jobs
        self.jobs = []
        self._sleep = None

    def start(self, sleep, start_delay=0):
        self._sleep = sleep
        sleep(start_delay, self._slot_free)

    def _job_done(self, job):
        self.jobs.remove(job)
        if self.free_slots == 1:
            self._slot_free()

    def _slot_free(self):
        job = self._get_job()
        if job:
            self.jobs.append(job)
            self._sleep(job, self._job_done, job)
        if self.free_slots:
            self._sleep(self._ttn(job is not None), self._slot_free)

    @property
    def n_jobs(self):
        return len(self.jobs)

    @property
    def free_slots(self):
        return self._max - self.n_jobs


n_workers = 3

def _rand_around(n):
    return n * random.gammavariate(5, 1) / 4

class QueueSim:
    def __init__(self, drop_freq=50, drop_size=30):
        self._drop_freq = drop_freq
        self._drop_size = drop_size
        self._sleep = None
        self._jobs = []
        self._job_length = 30
        self.n_gets = 0

    def start(self, sleep):
        self._sleep = sleep
        sleep(1, self._drop)

    def get_job(self):
        self.n_gets += 1
        if self._jobs:
            return self._jobs.pop()

    def _drop(self):
        self._jobs.extend(_rand_around(self._job_length) for _ in range(int(
            _rand_around(self._drop_size))))
        self._sleep(_rand_around(self._drop_freq), self._drop)

    @property
    def size(self):
        return len(self._jobs)


class ExistingAlgorithm:
    def __call__(self, got_job):
        if got_job:
            return 0.001
        return 1


class HalveLastAlgorithm:
    def __call__(self, got_job):
        if got_job:
            return 0.5
        return 1


class DoubleDelayed:
    def __init__(self):
        self._found_last = False

    def __call__(self, got_job):
        last2 = got_job and self.found_last
        self.found_last = got_job
        if last2:
            return 0.5
        return 1


class EvenAlgorithm:
    def __call__(self, got_job):
        return 1


class RandomAlgorithm:
    def __call__(self, got_job):
        return random.random() * 2


class BalancedAlgorithm:
    def __call__(self, got_job):
        return 0.5 + (self.worker.n_jobs/self.worker._max)


class StepBalanceAlgorithm:
    def __call__(self, got_job):
        v = 0.5 if got_job else 1
        return v * (1 + (self.worker.n_jobs/self.worker._max))


for algorithm in (ExistingAlgorithm, EvenAlgorithm, HalveLastAlgorithm, DoubleDelayed, BalancedAlgorithm, StepBalanceAlgorithm):
    print(algorithm.__name__)
    queue = QueueSim()
    algorithms = [algorithm() for _ in range(n_workers)]
    workers = [Worker(queue.get_job, a) for a in algorithms]
    for w, a in zip(workers, algorithms):
        a.worker = w

    class Sampler:
        def __init__(self):
            self.wasted_time = 0
            self.range_time = 0

        def summarise(self):
            print('Wasted time:', self.wasted_time)
            print('Average range:', self.range_time / (s.time - 1))
            print('Queue gets:', queue.n_gets)

        def sample(self, start, end):
            delta_t = end - start
            jobs = [w.n_jobs for w in workers]
#            print(max(jobs) - min(jobs))
            self.range_time += (max(jobs) - min(jobs)) * delta_t
            free_slots = sum(w.free_slots for w in workers)
            self.wasted_time += delta_t * min(queue.size, free_slots)

    sampler = Sampler()
    s = Scheduler(sampler.sample)
    queue.start(s.sleep)
    for worker in workers:
        worker.start(s.sleep, random.random())
    s.run_for(15000)
    sampler.summarise()
