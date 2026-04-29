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


def test_map_legend_swaps_to_upload_badge_during_preview():
    """Finding 2 from 2026-04-29-create-model-followups.md: while
    _uploaded_preview is set, the map legend should render a simple
    "Upload preview (N cells)" badge — not the production
    "Hexagonal Grid (185,428 cells)" label keyed on sim.mesh.n_cells.
    """
    from pathlib import Path

    app_py = (Path(__file__).resolve().parent.parent / "app.py").read_text(
        encoding="utf-8"
    )
    lines = app_py.splitlines()
    start = next(
        (i for i, ln in enumerate(lines)
         if "def map_legend(" in ln),
        None,
    )
    assert start is not None, "map_legend() reactive not found in app.py"
    end = next(
        (i for i in range(start + 1, min(start + 100, len(lines)))
         if lines[i].lstrip().startswith(("def ", "@", "# ---"))),
        len(lines),
    )
    fn_body = "\n".join(lines[start:end])
    assert "_uploaded_preview()" in fn_body, (
        "map_legend() must consult _uploaded_preview to swap legends; the "
        "production cell-count is misleading while the upload preview is "
        "active (Finding 2 in the followups doc)."
    )
    assert "Upload preview" in fn_body, (
        'map_legend() should emit an "Upload preview (N cells)" badge while '
        "an upload is rendered."
    )


def test_upload_preview_layer_id_is_namespaced():
    """Finding 1 from 2026-04-29-create-model-followups.md: the
    upload-preview H3HexagonLayer id must NOT collide with the production
    landscape's "water" layer. Production uses h3_hexagon_layer("water", ...)
    AND ScatterplotLayer({id: "water"}) for different landscape backends;
    deck.gl 9.2.10 cannot morph a layer's class under a stable id and emits
    shaderInputs draw errors on landscape switch. Namespace the upload
    layer id (e.g. "upload-water") to prevent the collision.
    """
    import re
    from pathlib import Path

    app_py = (Path(__file__).resolve().parent.parent / "app.py").read_text(
        encoding="utf-8"
    )
    lines = app_py.splitlines()
    start = next(
        (i for i, ln in enumerate(lines)
         if ln.startswith("def _build_preview_h3_layer")),
        None,
    )
    assert start is not None, "_build_preview_h3_layer not found in app.py"
    end = next(
        (i for i in range(start + 1, len(lines))
         if lines[i].startswith(("def ", "class "))),
        len(lines),
    )
    fn_body = "\n".join(lines[start:end])
    first_h3_call = re.search(r'h3_hexagon_layer\(\s*"([^"]+)"', fn_body)
    assert first_h3_call, "h3_hexagon_layer call not found in _build_preview_h3_layer"
    layer_id = first_h3_call.group(1)
    assert layer_id != "water", (
        f'Upload-preview H3HexagonLayer id is "{layer_id}" — collides with '
        f'the production landscape\'s "water" layer. Namespace it (e.g. '
        f'"upload-water") to prevent deck.gl shaderInputs draw errors on '
        f'landscape switch.'
    )
    assert "upload" in layer_id, (
        f"Upload-preview layer id should be namespaced with 'upload' prefix; "
        f"got '{layer_id}'"
    )
