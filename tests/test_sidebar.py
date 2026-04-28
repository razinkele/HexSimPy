"""Wiring tests for the sidebar UI module."""


def _render(sidebar):
    """Shiny's Sidebar repr is opaque; .tagify() returns the rendered TagList."""
    return str(sidebar.tagify())


def test_sidebar_includes_create_model_accordion():
    from ui.sidebar import sidebar_panel
    rendered = _render(sidebar_panel())
    assert "Create Model" in rendered
    assert "create_model_file" in rendered


def test_sidebar_create_model_inputs_have_expected_ids():
    from ui.sidebar import sidebar_panel
    rendered = _render(sidebar_panel())
    for input_id in (
        "create_model_file",
        "create_model_resolution",
        "create_model_with_bathy",
        "create_model_preview_btn",
        "create_model_status",
    ):
        assert input_id in rendered, f"missing input id: {input_id}"
