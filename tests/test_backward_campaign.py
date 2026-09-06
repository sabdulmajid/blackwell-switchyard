"""CPU-only regression checks for the unattended campaign contract."""

from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_backward_gpu_campaign.sh"
).read_text()


def test_campaign_reuses_only_valid_complete_phases():
    assert 'run_dir="$campaign_dir/run"' in SCRIPT
    assert "scripts/check_backward_report.py" in SCRIPT
    assert 'rm -f "$full_bf16"' in SCRIPT


def test_campaign_rejects_nonterminal_evaluator_results():
    assert "DROP|REJECT|READY_FOR_DISPATCH_REVIEW" in SCRIPT
    assert "did not reach a terminal decision" in SCRIPT


def test_campaign_audits_identity_and_committed_content():
    assert "GIT_AUTHOR_NAME=Ayman" in SCRIPT
    assert "GIT_AUTHOR_EMAIL=ayman.hasib@outlook.com" in SCRIPT
    assert "co-authored-by|claude|anthropic|wizchem|chatgpt" in SCRIPT
    assert "git grep -IinE" in SCRIPT
