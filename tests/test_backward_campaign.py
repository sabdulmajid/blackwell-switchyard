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
    assert "statistical instability requires fresh full evidence" in SCRIPT
    assert "deterministic incomplete evidence contract" in SCRIPT


def test_campaign_audits_identity_and_committed_content():
    assert "GIT_AUTHOR_NAME=Ayman" in SCRIPT
    assert "GIT_AUTHOR_EMAIL=ayman.hasib@outlook.com" in SCRIPT
    assert "co-authored-by|claude|anthropic|wizchem|chatgpt" in SCRIPT
    assert "git grep -IinE" in SCRIPT


def test_campaign_binds_decisions_and_copies_exact_bytes():
    assert "scripts/check_evidence_binding.py" in SCRIPT
    assert "evidence_hashes" in SCRIPT
    assert "copied evidence does not match its evaluated bytes" in SCRIPT


def test_campaign_preserves_per_attempt_guard_attestations():
    assert 'guard_attestation="$campaign_dir/guard.json"' in SCRIPT
    assert '"attempts": []' in SCRIPT
    assert 'document["attempts"].append(attempt)' in SCRIPT
    assert '"guard_attestations": guards' in SCRIPT
    assert 'key != "target_gpu_uuid"' in SCRIPT
    assert "tempfile.NamedTemporaryFile" in SCRIPT
    assert "os.fsync(handle.fileno())" in SCRIPT
    assert 'temporary = f"{path}.tmp"' not in SCRIPT


def test_campaign_publishes_only_to_exact_origin_and_recovers_pushes():
    assert "https://github.com/sabdulmajid/blackwell-switchyard.git" in SCRIPT
    assert "git remote get-url --push --all origin" in SCRIPT
    assert "push_result_commit" in SCRIPT
    assert "Recover a fully audited local result commit" in SCRIPT
    assert 'check_result_bundle "$result_prefix" :' in SCRIPT
    assert 'check_result_bundle "$result_prefix" HEAD' in SCRIPT
    assert "exit 74" in SCRIPT


def test_campaign_never_publishes_a_physical_gpu_uuid():
    assert "SWITCHYARD_PUBLIC_DEVICE_ID" in SCRIPT
    assert '"device_id": os.environ["SWITCHYARD_PUBLIC_DEVICE_ID"]' in SCRIPT
    assert "GPU-[[:alnum:]-]+" in SCRIPT
