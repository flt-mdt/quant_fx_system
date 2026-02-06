# API Contract

## QuantState JSON (initial contract)

```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "regime": {
    "label": "trend",
    "probability": 0.72
  },
  "signals": [
    {
      "name": "momentum",
      "value": 0.15
    }
  ],
  "decision": {
    "action": "long",
    "size": 0.4,
    "reason": "placeholder"
  },
  "meta": {
    "confidence": 0.63,
    "notes": "placeholder"
  }
}
```

Planned endpoints:
- `GET /health`
- `GET /state/latest`
- `GET /state/history?from=...&to=...`
- `POST /backtest/run`
- `GET /backtest/{id}`
