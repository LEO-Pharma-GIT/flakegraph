"""Tests for conservative extraction-window bibliography detection."""

from kg_processor.application.bibliography import is_reference_text


def test_reference_entries_require_multiple_independent_signals() -> None:
    """Recognize a real bibliography while retaining an isolated prose mention."""

    references = """References
[1] A. Author. First paper. 2020.
[2] B. Author. Second paper. 2021.
"""

    assert is_reference_text(references) is True
    assert is_reference_text("References\nThis section discusses reference architectures.") is False


def test_early_references_do_not_classify_later_primary_content() -> None:
    """Require each window to look like references instead of truncating a document."""

    preface_references = """References
[1] A. Editor. Note. 1987.
[2] B. Editor. Commentary. 1988.
"""
    later_paper = """A Complete Primary Research Paper
The proposed method and experiments begin here.
Results and discussion continue for many pages.
"""

    assert is_reference_text(preface_references) is True
    assert is_reference_text(later_paper) is False


def test_numbered_prose_list_is_not_a_bibliography() -> None:
    content = """Our contributions are:
1. We propose a bounded extraction method.
2. We evaluate it on public benchmarks.
3. We release code and checkpoints.
4. We analyze the observed failure modes.
"""

    assert is_reference_text(content) is False

    labeled_steps = """Implementation phases:
1. Phase A introduces the data model.
2. Phase B validates the data model.
3. Phase C deploys the model.
4. Phase D monitors the model.
"""
    assert is_reference_text(labeled_steps) is False


def test_vancouver_references_with_space_separated_initials_are_detected() -> None:
    content = """References
1. Smith J, Doe A. Effects of X on Y. Nature. 2020;12:45-60.
2. Garcia ML, Chen P. Follow-up study. Cell. 2021;9:10-20.
3. Okoro N, Adeyemi B. Longitudinal cohort study. Nature. 2022;9:300-320.
4. Nielsen K, Hansen TR. Replication results. Cell. 2023;4:15-25.
"""

    assert is_reference_text(content) is True
