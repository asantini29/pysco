from pysco.plots import journals


def test_packaged_style_constants_point_to_expected_names():
    assert journals.paper_style.endswith("mplfiles/paper.mplstyle")
    assert journals.corner_style.endswith("mplfiles/corner.mplstyle")


def test_get_style_resolves_packaged_style_path():
    style, params = journals.get_style("paper", journal="prd", cols="onecol")

    assert style.endswith("mplfiles/paper.mplstyle")
    assert params["figure.figsize"][0] == journals.journal_sizes["prd"]["onecol"]


def test_get_style_keeps_nonexistent_explicit_mplstyle_path(tmp_path):
    explicit = str(tmp_path / "custom.mplstyle")
    style, _ = journals.get_style(explicit)
    assert style == explicit


def test_get_style_keeps_existing_explicit_mplstyle_path(tmp_path):
    explicit_file = tmp_path / "custom.mplstyle"
    explicit_file.touch()
    style, _ = journals.get_style(str(explicit_file))
    assert style == str(explicit_file)
