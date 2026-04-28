# Infrastructure Anomaly Detector
### SentnelOps Internship Assignment — AI/ML Reasoning + Anomaly Detection

---

## What This Does

This is a Python-based system that takes infrastructure resource metrics (CPU, memory, network usage, and some security flags) and figures out whether a resource is behaving abnormally — and more importantly, *why*, and *what to do about it*.

The output for each resource is a structured JSON object with an anomaly classification, a human-readable explanation, a suggested action, a confidence score, and any security concerns worth flagging.

---

## My Approach

Before writing any code I spent some time thinking about what approach would actually make sense here. The three obvious options were: pure rule-based logic, an ML model like Isolation Forest, or just throwing everything at an LLM. I ended up going with a hybrid of all three ideas — though leaning heavily on rules for the actual detection.

Here's the rough pipeline:

```
Raw Metrics JSON
      │
      ▼
  Rule Engine          →  figures out what kind of anomaly this is
      │
      ▼
  Confidence Scorer    →  computes how confident we should be (0.0–1.0)
      │
      ▼
  Security Check       →  independent check on internet exposure + IAM
      │
      ▼
  LLM Reasoner         →  generates a natural language explanation
      │
      ▼
  Structured JSON Output
```

### Why rules for detection?

The metrics here are small, structured, and domain-specific. CPU, memory, and network all have well-understood normal ranges. Given that, hard-coded thresholds aren't a limitation — they're actually the right tool. They're transparent, auditable, and produce the same output every time for the same input.

I also considered Isolation Forest, but it needs a reasonably sized dataset to learn a meaningful distribution from. With 7 data points and clear domain knowledge already in hand, training an unsupervised model would've been more theater than substance. That said, I do think ML becomes the right call at scale — if you're ingesting 30 days of 5-minute CPU samples across thousands of instances, you'd want something that can detect gradual drift or seasonal patterns that rules would miss.

### Why not just use an LLM for everything?

An LLM-only approach would work reasonably well for explanation quality, but it's non-deterministic, slower, and you can't reliably control what anomaly type it assigns or how it computes confidence. The facts need to come from somewhere grounded first — the LLM's job here is to turn structured findings into a clear explanation, not to do the analysis itself. That way there's no hallucination risk in the detection layer.

The LLM call (Claude via the Anthropic API) is also completely optional. If no API key is set, the system falls back to rule-based explanation templates and still produces clean, complete output.

---

## Anomaly Types

The rule engine can detect seven different anomaly types, and a single resource can trigger more than one:

**`over_provisioned`** — CPU average and peak are both very low, meaning the instance is much larger than the workload actually needs. Classic cloud waste.

**`idle_resource`** — Both CPU and memory are near zero. The instance is running but probably doing nothing. Could be a forgotten dev box, a failed job that left the machine up, or a misconfigured autoscaler.

**`cpu_spike`** — Average CPU looks fine, but the p95 is very high. This is the tricky one that aggregate metrics hide. The workload is bursty — fine most of the time, then slammed. Standard dashboards looking at averages would miss this entirely.

**`cpu_saturation`** — CPU is consistently high across both average and peak. The instance is genuinely overloaded and probably already causing slowdowns.

**`memory_pressure`** — Memory usage is critically high. Risk of OOM kills, swap usage, and cascading failures if the workload grows at all.

**`network_anomaly`** — Network bandwidth is unusually high relative to what CPU and memory suggest the workload should be doing. Could be legitimate (a data pipeline, backup job) but worth flagging, especially on internet-facing machines.

**`hot_resource`** — CPU and memory are both maxed out simultaneously. Highest severity — the instance is at risk of full failure.

---

## Confidence Scoring

One thing I wanted to avoid was just hardcoding confidence values like `0.78` for a given rule. That's not meaningful. Instead, confidence is computed as a function of how far the triggering metric deviates from its threshold.

For example, if the threshold for `over_provisioned` is a CPU average below 10%, a resource at 9% sits right on the edge — the system isn't very sure. A resource at 1% CPU is clearly, deeply over-provisioned — the system should be much more confident. The score reflects that proportionally.

Multiple anomalies firing together also compounds the confidence upward, and security flags add a small bump because a flagged resource warrants higher urgency even if the metrics are ambiguous. The final score is capped at 0.99 — I didn't want to express 100% certainty from a system like this.

---

## Security Evaluation

Security is handled as a completely separate check from anomaly detection. The logic is a simple 2-factor matrix: is the resource internet-facing, and does it have an identity (IAM role / service account) attached?

An internet-facing machine with an identity attached is the highest concern — it's both publicly reachable and carries permissions that could be exploited. An idle or over-provisioned machine in this state is especially worth flagging because it's an unnecessary attack surface with no business justification for being up.

The security note is always reported independently in the output, so a healthy resource can still surface a security concern, and a badly performing resource with no public exposure won't have its security section inflated.

---

## Running It

The only dependency is `requests`. Everything else is standard library.

```bash
pip install requests
```

Run on the built-in test cases (7 resources covering every anomaly type):

```bash
python anomaly_detector.py
```

Run on your own JSON file:

```bash
python anomaly_detector.py my_resources.json
```

To enable LLM-generated explanations via Claude:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python anomaly_detector.py
```

Without the key it falls back gracefully — you still get full output, just with rule-based explanations instead of LLM-generated ones.

Input should be a JSON array of resource objects like this:

```json
[
  {
    "resource_id": "i-1",
    "cpu_avg": 2,
    "cpu_p95": 5,
    "memory_avg": 70,
    "network_pct": 10,
    "internet_facing": true,
    "identity_attached": true
  }
]
```

Results are printed to stdout and also saved to `sample_outputs.json`.

---

## Output Format

Each resource produces one JSON object:

```json
{
  "resource_id": "i-3",
  "is_anomalous": true,
  "anomaly_types": ["cpu_spike"],
  "primary_anomaly": "cpu_spike",
  "reason": "Average CPU is low but peak (p95) is very high, indicating a bursty workload. Standard avg metrics would mask these bursts entirely.",
  "suggested_action": "Consider auto-scaling policies or a burstable instance type designed for spiky workloads.",
  "confidence": 0.69,
  "security_note": "MEDIUM RISK — Internal resource with identity/IAM attached. Ensure least-privilege policies are enforced."
}
```

`anomaly_types` is a list because a resource can have multiple issues. `primary_anomaly` is whichever fired first and leads the explanation. `security_note` is `null` if there's no concern.

---

## Tradeoffs

The main tradeoff in going hybrid is complexity — there are three layers to maintain instead of one. The LLM call also adds latency (usually 1–3 seconds per resource) and an external dependency.

On the other side: the rule engine alone would produce rigid, repetitive explanations that don't adapt to context. The LLM alone would be slow, expensive, and unpredictable for classification. Putting them together gets you deterministic detection with natural language output, which I think is the right tradeoff for something meant to be read and acted on by a human ops team.

The thresholds I picked (CPU < 10% for over-provisioned, > 80% for saturation, etc.) are reasonable defaults but would ideally be tuned per environment. A compute-intensive batch workload has a very different "normal" than a web API server. In a real deployment you'd want these to be configurable per resource tag or workload type.

---

## What I'd Do Differently With More Time

The thing I'd want to add most is time-series support. Right now the system works on a snapshot — a single set of metrics at one point in time. Real anomalies often show up as trends: CPU gradually climbing over 6 hours, memory that never drops below 90% after a deployment, network that spikes every night at 2am. None of that is detectable from a snapshot.

I'd also add a feedback mechanism — a field in the output where an ops engineer can mark whether a detection was a true or false positive. Over time that feedback could be used to automatically adjust thresholds toward the values that actually matter for that environment.

Other things I'd look at: severity-weighted sorting so the most critical resources show up first, a simple web UI for visualizing the output across a fleet, and a proper logging layer so you can track how anomaly patterns change over time.

---

## Files

```
.
├── anomaly_detector.py   # Main script
├── sample_outputs.json   # Output from 7 test cases
└── README.md             # This file
```
