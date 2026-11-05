import math
from src.epistemic.belief_graph import BeliefGraph, logit, sigmoid


def ev(status: str, sup=(), con=(), conflict: bool = False):
    return {
        "status": status,
        "conflict": conflict,
        "sup": list(sup),
        "con": list(con),
    }


def test_support_raises_and_contradiction_lowers():
    g = BeliefGraph(w_s=1.0, w_c=2.0)
    g.add("Claim A", True, [], 0.5, ev("supported", sup=[("Doc 1", 0.9)]))
    g.add("Claim B", False, [], 0.5, ev("contradicted", con=[("Doc 2", 0.9)]))
    g.update()

    # Support should raise confidence above prior (0.5), contradiction should lower it
    assert g.claims[0].conf > 0.5
    assert g.claims[1].conf < 0.5
    assert g.claims[0].conf > g.claims[1].conf


def test_clipping():
    # Large evidence weights should be clipped to [-clip, clip] = [-6, 6]
    g = BeliefGraph(w_s=100.0, w_c=1.0, clip=6.0)
    g.add("Claim A", True, [], 0.9, ev("supported", sup=[("Doc 1", 1.0)]))
    g.update()

    expected_conf = sigmoid(6.0)
    assert abs(g.claims[0].conf - expected_conf) < 1e-9


def test_weakest_link_and_switch():
    def build(prop: bool) -> BeliefGraph:
        g = BeliefGraph(w_s=1.0, w_c=1.0, propagate=prop)
        # Claim A has low confidence (0.2), unresolved
        g.add("Claim A", False, [], 0.2, ev("unresolved"))
        # Claim B depends on Claim A, has high evidence support
        g.add("Claim B", True, [0], 0.9, ev("supported", sup=[("Doc 1", 0.9)]))
        return g.update()

    # With propagation: Claim B's effective confidence cannot exceed parent Claim A
    g_prop = build(True)
    assert g_prop.claims[1].eff <= g_prop.claims[0].eff + 1e-9

    # Without propagation: Claim B's effective confidence reflects its own strong support
    g_no_prop = build(False)
    assert g_no_prop.claims[1].eff > g_prop.claims[1].eff


def test_forward_dependencies_are_ignored():
    g = BeliefGraph()
    # Adding a dependency index >= current index (e.g. forward reference 5) must be discarded
    g.add("Claim A", True, [5], 0.5, ev("unresolved"))
    g.update()
    assert g.claims[0].depends_on == []


def test_gaps_identification():
    g = BeliefGraph()
    g.add("Claim A", False, [], 0.3, ev("unresolved"))
    g.add("Claim B", True, [], 0.9, ev("supported", sup=[("Doc 1", 0.9)]))
    g.update()

    gaps = g.gaps(theta_gap=0.6, use_status=True)
    assert len(gaps) == 1
    assert gaps[0].text == "Claim A"
    assert g.weakest(gaps).text == "Claim A"
