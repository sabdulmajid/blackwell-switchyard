"""CPU-only regression checks for the unattended campaign contract."""

from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_backward_gpu_campaign.sh"
).read_text()


def test_campaign_reuses_only_valid_complete_phases():
    assert 'run_dir="$campaign_dir/run"' in SCRIPT
    assert "scripts/check_backward_report.py" in SCRIPT
    assert 'rm -f "$full_bf16"' in SCRIPT
    assert '--expected-tree "$expected_tree"' in SCRIPT


def test_campaign_rejects_nonterminal_evaluator_results():
    assert "0:DROP|0:READY_FOR_DISPATCH_REVIEW|1:REJECT" in SCRIPT
    assert "2:MORE_DATA" in SCRIPT
    assert "evaluator requested fresh full evidence" in SCRIPT


def test_campaign_audits_identity_and_committed_content():
    assert "GIT_AUTHOR_NAME=Ayman" in SCRIPT
    assert "GIT_AUTHOR_EMAIL=ayman.hasib@outlook.com" in SCRIPT
    assert "co-authored-by|claude|anthropic|wizchem|chatgpt" in SCRIPT
    assert "git grep -IinE" in SCRIPT


def test_campaign_binds_decisions_and_copies_exact_bytes():
    assert "scripts/check_evidence_binding.py" in SCRIPT
    assert "evidence_hashes" in SCRIPT
    assert "copied evidence does not match its evaluated bytes" in SCRIPT


def test_campaign_publishes_only_to_exact_origin_and_recovers_pushes():
    assert "https://github.com/sabdulmajid/blackwell-switchyard.git" in SCRIPT
    assert "git remote get-url --push origin" in SCRIPT
    assert "push_result_commit" in SCRIPT
    assert "Recover a fully audited local result commit" in SCRIPT
