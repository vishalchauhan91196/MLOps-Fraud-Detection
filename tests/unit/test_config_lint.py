from src.core.config_lint import lint_all_environments


def test_config_lint_passes_for_all_envs():
    """Validate that config lint passes for all envs behaves as expected.

    This test guards against regressions in the associated workflow or API behavior.
    """
    ok, errors = lint_all_environments()
    assert ok is True
    assert errors == []

