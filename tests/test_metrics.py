def test_confidence_threshold():
    from metrics import compute_metrics
    metrics = compute_metrics()
    assert metrics["avg_conf"] >= 0.80, "Alerte: score moyen < seuil"
