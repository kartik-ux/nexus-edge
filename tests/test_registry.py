from nexus_edge.data import make_dataset
from nexus_edge.model import fit_model
from nexus_edge.registry import ModelRegistry

def test_registry_promote_and_rollback(tmp_path):
    data = make_dataset(300, 60)
    model = fit_model(data)
    registry = ModelRegistry(tmp_path)
    first = registry.promote(model, {"f1": .8})
    second = registry.promote(model, {"f1": .81})
    assert registry.active_version() == second
    registry.rollback(first)
    assert registry.active_version() == first
    assert registry.load_active() is not None
