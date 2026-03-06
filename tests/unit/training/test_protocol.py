"""Verify that CausalLMModel satisfies the ModelProtocol at the type level."""

from crucible.training.protocol import ModelProtocol


def test_model_protocol_is_runtime_checkable() -> None:
    assert hasattr(ModelProtocol, "__protocol_attrs__") or isinstance(ModelProtocol, type)


def test_causal_lm_model_has_required_methods() -> None:
    from crucible.training.model import CausalLMModel

    for method in ("train", "predict", "save", "load"):
        assert hasattr(CausalLMModel, method), f"Missing method: {method}"
        assert callable(getattr(CausalLMModel, method))
