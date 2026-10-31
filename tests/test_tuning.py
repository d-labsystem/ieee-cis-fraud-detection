import pytest

from fraud.tuning import config_name, parse_params


def test_parse_params_uses_default_types():
    params = parse_params(["learning_rate=0.1", "max_depth=6", "enable_categorical=False"])
    assert params == {"learning_rate": 0.1, "max_depth": 6, "enable_categorical": False}
    assert isinstance(params["max_depth"], int)


def test_parse_params_unknown_keys():
    params = parse_params(["min_child_weight=5", "gamma=0.5", "grow_policy=lossguide"])
    assert params == {"min_child_weight": 5, "gamma": 0.5, "grow_policy": "lossguide"}


@pytest.mark.parametrize("item", ["max_depth", "=6", "max_depth=", "enable_categorical=yes"])
def test_parse_params_rejects_bad_items(item):
    with pytest.raises(ValueError):
        parse_params([item])


def test_config_name():
    assert config_name("client+entity", {}) == "client+entity"
    assert (config_name("client+entity", {"max_depth": 6, "learning_rate": 0.1})
            == "client+entity@learning_rate=0.1,max_depth=6")
