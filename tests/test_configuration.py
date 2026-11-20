import json
from pathlib import Path

import yaml

ROOT=Path(__file__).resolve().parents[1]


def test_compose_services_and_monitoring():
    data=yaml.safe_load((ROOT/'compose.yaml').read_text())
    assert {'api','db','prometheus','grafana','mlflow'} <= set(data['services'])
    assert 'DB_URL' in data['services']['api']['environment']
    assert data['services']['api']['read_only']
    prom=yaml.safe_load((ROOT/'monitoring/prometheus/prometheus.yml').read_text())
    assert prom['scrape_configs'][0]['metrics_path']=='/metrics'
    alerts=yaml.safe_load((ROOT/'monitoring/prometheus/alerts.yml').read_text())
    assert 'toxicity_http_requests_total' in str(alerts)
    dashboard=json.loads((ROOT/'monitoring/grafana/dashboards/toxicity-overview.json').read_text())
    assert dashboard['panels']


def test_isolated_postgres_for_integration():
    data=yaml.safe_load((ROOT/'compose.yaml').read_text())
    testdb=data['services']['db_test']
    assert 'test' in testdb['profiles']
    assert testdb['environment']['POSTGRES_DB']=='toxicity_test'
    assert testdb['ports'] == ['127.0.0.1:55432:5432']
    assert 'volumes' not in testdb
    assert '/var/lib/postgresql/data' in testdb['tmpfs']
