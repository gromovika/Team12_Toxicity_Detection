"""Prometheus counters, no raw texts or identifiers in metric labels."""
from prometheus_client import Counter, Histogram

PREDICTIONS = Counter('toxicity_predictions_total', 'Successful classifications',
                      ['source', 'predicted_class'])
INFERENCE_LATENCY = Histogram('toxicity_inference_duration_seconds',
                              'Duration of model inference', ['source'],
                              buckets=(.005,.01,.025,.05,.1,.25,.5,1,2,5))
HTTP_REQUESTS = Counter('toxicity_http_requests_total', 'HTTP responses by route and status',
                        ['route','status'])
HTTP_LATENCY = Histogram('toxicity_http_request_duration_seconds', 'Request duration',
                         ['route'], buckets=(.005,.01,.025,.05,.1,.25,.5,1,2,5))
