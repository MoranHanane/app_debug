import time
from collections import deque

# Stockage des métriques
request_times = deque(maxlen=100)
pred_confidences = deque(maxlen=100)
pred_classes = deque(maxlen=100)

def log_request_time(duration): #log de latence
    request_times.append(duration)

def log_prediction(cls, score): #log de classification + score de confiance
    pred_classes.append(cls)
    pred_confidences.append(score)

def compute_metrics():
    total_requests = len(request_times)
    total_predictions = len(pred_confidences)

    if total_requests == 0:
        avg_latency = 0
        error_rate = 0
    else:
        avg_latency = sum(request_times) / total_requests
        error_rate = sum(1 for c in request_times if c > 1) / total_requests    #error rate si latence > 1seconde

    if total_predictions == 0:
        avg_conf = 0
        class_distribution = {}
    else:
        avg_conf = sum(pred_confidences) / total_predictions
        class_distribution = {
            c: pred_classes.count(c) for c in set(pred_classes)
        }

    return {
        "avg_latency": avg_latency,
        "error_rate": error_rate,
        "avg_conf": avg_conf,
        "class_distribution": class_distribution,
        "total_predictions": total_predictions
    }