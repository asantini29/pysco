from pathlib import Path

from pysco.plots import journals


def test_packaged_styles_exist():
    assert Path(journals.paper_style).is_file()
    assert Path(journals.corner_style).is_file()


def test_get_style_resolves_packaged_style_path():
    style, params = journals.get_style("paper", journal="prd", cols="onecol")

    assert Path(style).is_file()
    assert Path(style).name == "paper.mplstyle"
    assert params["figure.figsize"][0] == journals.journal_sizes["prd"]["onecol"]


def test_get_style_keeps_explicit_mplstyle_path():
    explicit = "/tmp/custom.mplstyle"
    style, _ = journals.get_style(explicit)
    assert style == explicit
