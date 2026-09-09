from src.agents.supervisor import Supervisor


class DummyAgent:
    def __init__(self):
        self.received = []

    def handle_task(self, task):
        self.received.append(task)
        return {'ok': True, 'task': task}


def test_supervisor_register_and_dispatch():
    sup = Supervisor()
    agent = DummyAgent()
    sup.register('d', agent)
    res = sup.dispatch('d', {'type': 'echo', 'payload': 123})
    assert res['ok'] is True
    assert agent.received[0]['payload'] == 123


def test_supervisor_broadcast():
    sup = Supervisor()
    a1 = DummyAgent()
    a2 = DummyAgent()
    sup.register('a1', a1)
    sup.register('a2', a2)
    res = sup.broadcast({'type': 'ping'})
    assert 'a1' in res and 'a2' in res
    assert res['a1']['ok'] is True
