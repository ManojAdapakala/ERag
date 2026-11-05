from src.llm.client import LLM


def make():
    l = LLM.__new__(LLM)  # skips API client, so test needs no key
    l.m = {"name": "m", "reasoning_effort": "low", "max_tokens": 10}
    l.namespace = "ns"
    return l


def test_key_changes_with_every_setting():
    l, msgs = make(), [{"role": "user", "content": "hi"}]
    base = l._key(msgs, 0.0, 10, 0)

    # 1. Temperature change alters key
    assert base != l._key(msgs, 0.7, 10, 0)

    # 2. Max tokens change alters key
    assert base != l._key(msgs, 0.0, 11, 0)

    # 3. Sample index change alters key (critical for self-consistency)
    assert base != l._key(msgs, 0.0, 10, 1)

    # 4. Prompt content alters key
    assert base != l._key([{"role": "user", "content": "hello"}], 0.0, 10, 0)

    # 5. Model name alters key
    l.m = dict(l.m, name="other")
    assert base != l._key(msgs, 0.0, 10, 0)

    # 6. Namespace alters key
    l.m = dict(l.m, name="m")
    l.namespace = "ns2"
    assert base != l._key(msgs, 0.0, 10, 0)

    # 7. Reasoning effort alters key
    l.namespace = "ns"
    l.m = dict(l.m, reasoning_effort="high")
    assert base != l._key(msgs, 0.0, 10, 0)

