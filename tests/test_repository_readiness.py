"""Contract checks for public repository docs and portable host port."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_docker_host_port_is_configurable_and_internal_target_unchanged():
    compose = yaml.safe_load((ROOT / 'compose.yaml').read_text(encoding='utf-8'))
    ports = compose['services']['api']['ports']
    assert '127.0.0.1:${API_HOST_PORT:-8000}:8000' in ports
    script = (ROOT / 'scripts/local_validate.sh').read_text(encoding='utf-8')
    assert 'API_BASE_URL="http://127.0.0.1:${API_HOST_PORT}"' in script
    assert '"$API_BASE_URL/health/ready"' in script
    assert '"$API_BASE_URL/predict"' in script
    assert '"$API_BASE_URL/metrics"' in script
    assert 'API_HOST_PORT=8000' in (ROOT / '.env.example').read_text(encoding='utf-8')


def test_review_documents_present_and_linked():
    required = ('ARCHITECTURE_RU.md', 'API_CONTRACT_RU.md',
                'DATA_CARD_RU.md', 'EXPERIMENTS_RU.md', 'TEST_PLAN_RU.md',
                'MONITORING_PLAN_RU.md',
                'SUBMISSION_CHECKLIST_RU.md', 'LOCAL_VERIFICATION_RU.md')
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    for doc in required:
        assert (ROOT / 'docs' / doc).is_file(), doc
    for doc in required[:8]:
        assert doc in readme or doc in (ROOT / 'docs' / 'SUBMISSION_CHECKLIST_RU.md').read_text(), doc
    gitignore = (ROOT / '.gitignore').read_text(encoding='utf-8')
    assert 'data/' in gitignore and '.env' in gitignore
