"""
SentnelOps Internship Assignment
AI/ML Reasoning + Anomaly Detection
Author: Hrishi
Approach: Hybrid (Rule Engine + Confidence Scorer + LLM Reasoner)
"""

import json
import os
import requests
from typing import Optional

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
USE_LLM = bool(ANTHROPIC_API_KEY)  # Falls back to rule-based reason if no key

# ─────────────────────────────────────────────────────────────
# THRESHOLDS
# ─────────────────────────────────────────────────────────────

THRESHOLDS = {
    "cpu_low": 10,         # below this → idle / over-provisioned
    "cpu_high": 80,        # above this → saturation
    "cpu_p95_high": 90,    # above this → peak saturation
    "cpu_spike_avg": 40,   # avg below this but p95 high → bursty
    "cpu_spike_p95": 80,   # p95 above this (combined with low avg) → spike
    "memory_high": 85,     # above this → memory pressure
    "memory_low": 20,      # below this (+ low cpu) → idle
    "network_high": 80,    # above this → network anomaly
    "cpu_very_low": 5,     # for idle ghost detection
}

# ─────────────────────────────────────────────────────────────
# LAYER 1: RULE ENGINE
# ─────────────────────────────────────────────────────────────

def detect_anomalies(resource: dict) -> list[str]:
    """
    Applies multi-signal rules to detect anomaly types.
    A resource can have multiple anomaly types.
    Returns a list of detected anomaly type strings.
    """
    cpu_avg     = resource.get("cpu_avg", 0)
    cpu_p95     = resource.get("cpu_p95", 0)
    memory_avg  = resource.get("memory_avg", 0)
    network_pct = resource.get("network_pct", 0)

    T = THRESHOLDS
    detected = []

    # Idle ghost: extremely low CPU and memory
    if cpu_avg < T["cpu_very_low"] and memory_avg < T["memory_low"]:
        detected.append("idle_resource")

    # Over-provisioned: low CPU but not necessarily low memory
    elif cpu_avg < T["cpu_low"] and cpu_p95 < 20:
        detected.append("over_provisioned")

    # CPU spike: low average but very high peak → bursty workload
    # Check BEFORE saturation — a low avg + high p95 is a spike, not sustained saturation
    if cpu_avg < T["cpu_spike_avg"] and cpu_p95 > T["cpu_spike_p95"]:
        detected.append("cpu_spike")

    # CPU saturation: consistently high CPU (avg must also be high to distinguish from spikes)
    elif cpu_avg > T["cpu_high"] or cpu_p95 > T["cpu_p95_high"]:
        if "cpu_spike" not in detected:
            detected.append("cpu_saturation")

    # Memory pressure
    if memory_avg > T["memory_high"]:
        detected.append("memory_pressure")

    # Network anomaly
    if network_pct > T["network_high"]:
        detected.append("network_anomaly")

    # Hot resource: both CPU and memory are maxed
    if cpu_avg > T["cpu_high"] and memory_avg > T["memory_high"]:
        if "cpu_saturation" not in detected:
            detected.append("hot_resource")
        if "memory_pressure" not in detected:
            detected.append("memory_pressure")

    return detected if detected else []


# ─────────────────────────────────────────────────────────────
# LAYER 2: CONFIDENCE SCORER
# ─────────────────────────────────────────────────────────────

def compute_confidence(resource: dict, anomaly_types: list[str]) -> float:
    """
    Computes a confidence score (0.0 – 1.0) based on:
    - How far each metric deviates from its threshold
    - Number of signals firing (more signals → higher confidence)
    - Security flags as a small bonus signal
    """
    if not anomaly_types:
        return 0.0

    T = THRESHOLDS
    cpu_avg     = resource.get("cpu_avg", 0)
    cpu_p95     = resource.get("cpu_p95", 0)
    memory_avg  = resource.get("memory_avg", 0)
    network_pct = resource.get("network_pct", 0)
    internet    = resource.get("internet_facing", False)
    identity    = resource.get("identity_attached", False)

    score = 0.45  # base score for any anomaly detection

    for atype in anomaly_types:
        if atype == "over_provisioned":
            # Deviation: how far below 10% cpu_avg is
            dev = max(0, (T["cpu_low"] - cpu_avg) / T["cpu_low"])
            score += dev * 0.3

        elif atype == "idle_resource":
            dev = max(0, (T["cpu_very_low"] - cpu_avg) / T["cpu_very_low"])
            score += dev * 0.35

        elif atype == "cpu_saturation":
            dev = max(0, (cpu_avg - T["cpu_high"]) / (100 - T["cpu_high"]))
            p95_dev = max(0, (cpu_p95 - T["cpu_p95_high"]) / (100 - T["cpu_p95_high"]))
            score += (dev * 0.2) + (p95_dev * 0.15)

        elif atype == "cpu_spike":
            # Bigger the gap between avg and p95, higher the spike confidence
            gap = max(0, cpu_p95 - cpu_avg) / 100
            score += gap * 0.3

        elif atype == "memory_pressure":
            dev = max(0, (memory_avg - T["memory_high"]) / (100 - T["memory_high"]))
            score += dev * 0.2

        elif atype == "network_anomaly":
            dev = max(0, (network_pct - T["network_high"]) / (100 - T["network_high"]))
            score += dev * 0.2

        elif atype == "hot_resource":
            score += 0.15  # Both CPU and memory maxed is inherently high confidence

    # Security amplifier: flagged resources should be reported with higher urgency
    if internet and identity:
        score += 0.05
    elif internet or identity:
        score += 0.02

    return round(min(score, 0.99), 2)


# ─────────────────────────────────────────────────────────────
# SECURITY RISK MATRIX
# ─────────────────────────────────────────────────────────────

def evaluate_security(resource: dict, anomaly_types: list[str]) -> Optional[str]:
    """
    2-factor security risk matrix based on:
    - internet_facing: is the resource publicly exposed?
    - identity_attached: does it have IAM/service account permissions?
    Returns a security note string, or None if no concern.
    """
    internet = resource.get("internet_facing", False)
    identity = resource.get("identity_attached", False)
    is_anomalous = bool(anomaly_types)

    if internet and identity:
        note = "HIGH RISK — Internet-facing resource with identity/IAM attached."
        if is_anomalous:
            note += f" Combined with detected anomaly ({', '.join(anomaly_types)}), this resource should be reviewed immediately."
        return note

    elif internet and not identity:
        note = "MEDIUM RISK — Resource is publicly exposed to the internet."
        if "idle_resource" in anomaly_types or "over_provisioned" in anomaly_types:
            note += " An idle internet-facing instance is an unnecessary attack surface — consider shutting down or firewalling."
        return note

    elif not internet and identity:
        return "MEDIUM RISK — Internal resource with identity/IAM attached. Ensure least-privilege policies are enforced."

    return None  # No security concern


# ─────────────────────────────────────────────────────────────
# RULE-BASED FALLBACK REASONS (when no LLM key)
# ─────────────────────────────────────────────────────────────

FALLBACK_REASONS = {
    "over_provisioned": (
        "CPU utilization is significantly below capacity at both average and peak levels, "
        "indicating the instance is larger than the workload requires.",
        "Downsize to a smaller instance type to reduce cost. Review memory needs before resizing."
    ),
    "idle_resource": (
        "Both CPU and memory usage are critically low, suggesting this resource is either "
        "unused or running an empty/stopped workload.",
        "Investigate if this instance is still needed. If not, terminate or hibernate to eliminate waste."
    ),
    "cpu_saturation": (
        "CPU is consistently near or at maximum capacity. This will cause performance degradation, "
        "timeouts, or dropped requests under load.",
        "Scale vertically (larger instance) or horizontally (add replicas). Profile the workload to identify hotspots."
    ),
    "cpu_spike": (
        "Average CPU is low but peak (p95) is very high, indicating a bursty workload pattern. "
        "Standard metrics may mask these bursts.",
        "Consider auto-scaling policies or a burstable instance type designed for spiky workloads."
    ),
    "memory_pressure": (
        "Memory utilization is critically high. This risks OOM kills, swap usage, "
        "and severe performance degradation.",
        "Increase memory allocation or investigate memory leaks in the running application."
    ),
    "network_anomaly": (
        "Network bandwidth consumption is unusually high. This may indicate data exfiltration, "
        "a misconfigured service, or a DDoS pattern.",
        "Inspect network flow logs immediately. Rate-limit or block suspicious traffic sources."
    ),
    "hot_resource": (
        "Both CPU and memory are simultaneously under extreme load. "
        "The resource is at risk of full exhaustion and system failure.",
        "Immediate action required: scale up resources and investigate root cause of combined pressure."
    ),
}

def get_fallback_reason(anomaly_types: list[str]) -> tuple[str, str]:
    """Returns (reason, suggested_action) from rule-based fallbacks."""
    if not anomaly_types:
        return (
            "All metrics are within normal operating ranges. No anomalies detected.",
            "No action required. Continue standard monitoring."
        )
    # Use the first (primary) anomaly type as the lead reason
    primary = anomaly_types[0]
    reason, action = FALLBACK_REASONS.get(primary, (
        f"Detected anomaly: {', '.join(anomaly_types)}. Metrics deviate from expected baselines.",
        "Investigate the flagged metrics and review resource sizing and security posture."
    ))
    if len(anomaly_types) > 1:
        extras = ", ".join(anomaly_types[1:])
        reason += f" Additionally detected: {extras}."
    return reason, action


# ─────────────────────────────────────────────────────────────
# LAYER 3: LLM REASONER (Claude API)
# ─────────────────────────────────────────────────────────────

def llm_reason(resource: dict, anomaly_types: list[str], security_note: Optional[str]) -> tuple[str, str]:
    """
    Calls Claude to generate a natural language reason and suggested action.
    Falls back to rule-based if API key is missing or call fails.
    """
    if not USE_LLM:
        return get_fallback_reason(anomaly_types)

    prompt = f"""You are an infrastructure anomaly analysis system. Analyze the following resource metrics and provide a brief, expert explanation.

Resource Metrics:
{json.dumps(resource, indent=2)}

Detected Anomaly Types: {anomaly_types if anomaly_types else ['none — resource appears healthy']}
Security Risk: {security_note or 'None'}

Instructions:
- Write a "reason" explaining WHY this resource is anomalous (or healthy) in exactly 2 sentences. Be specific about the metric values.
- Write a "suggested_action" that is concrete and actionable in 1 sentence.
- If no anomaly, say so clearly.

Respond ONLY with a valid JSON object. No preamble, no markdown, no backticks:
{{"reason": "...", "suggested_action": "..."}}"""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        text = data["content"][0]["text"].strip()
        parsed = json.loads(text)
        return parsed["reason"], parsed["suggested_action"]
    except Exception as e:
        print(f"  [LLM fallback] Claude API error: {e}")
        return get_fallback_reason(anomaly_types)


# ─────────────────────────────────────────────────────────────
# MAIN ANALYZER
# ─────────────────────────────────────────────────────────────

def analyze_resource(resource: dict) -> dict:
    """
    Full pipeline: Rule Engine → Scorer → Security → LLM Reasoner → Output
    """
    resource_id = resource.get("resource_id", "unknown")
    print(f"  Analyzing {resource_id}...")

    # Layer 1: Rules
    anomaly_types = detect_anomalies(resource)
    is_anomalous  = bool(anomaly_types)
    primary_type  = anomaly_types[0] if anomaly_types else "none"

    # Layer 2: Confidence
    confidence = compute_confidence(resource, anomaly_types)

    # Security Matrix
    security_note = evaluate_security(resource, anomaly_types)

    # Layer 3: LLM Reason
    reason, suggested_action = llm_reason(resource, anomaly_types, security_note)

    return {
        "resource_id":      resource_id,
        "is_anomalous":     is_anomalous,
        "anomaly_types":    anomaly_types,
        "primary_anomaly":  primary_type,
        "reason":           reason,
        "suggested_action": suggested_action,
        "confidence":       confidence,
        "security_note":    security_note,
    }


def analyze_batch(resources: list[dict]) -> list[dict]:
    print(f"\n{'='*55}")
    print(f"  SentnelOps Anomaly Detector — {len(resources)} resources")
    print(f"  LLM reasoning: {'ENABLED (Claude)' if USE_LLM else 'DISABLED (rule-based fallback)'}")
    print(f"{'='*55}\n")
    results = [analyze_resource(r) for r in resources]
    print(f"\n✅ Analysis complete.\n")
    return results


# ─────────────────────────────────────────────────────────────
# TEST DATA (7 cases)
# ─────────────────────────────────────────────────────────────

TEST_RESOURCES = [
    # From assignment
    {
        "resource_id": "i-1",
        "cpu_avg": 2, "cpu_p95": 5,
        "memory_avg": 70, "network_pct": 10,
        "internet_facing": True, "identity_attached": True
    },
    {
        "resource_id": "i-2",
        "cpu_avg": 85, "cpu_p95": 98,
        "memory_avg": 40, "network_pct": 60,
        "internet_facing": False, "identity_attached": False
    },
    # Additional test cases
    {
        "resource_id": "i-3",
        "cpu_avg": 18, "cpu_p95": 91,
        "memory_avg": 45, "network_pct": 30,
        "internet_facing": False, "identity_attached": True,
        "note": "Bursty workload — low avg, very high peak"
    },
    {
        "resource_id": "i-4",
        "cpu_avg": 88, "cpu_p95": 97,
        "memory_avg": 91, "network_pct": 55,
        "internet_facing": True, "identity_attached": True,
        "note": "Hot resource — CPU and memory both maxed"
    },
    {
        "resource_id": "i-5",
        "cpu_avg": 3, "cpu_p95": 4,
        "memory_avg": 8, "network_pct": 5,
        "internet_facing": True, "identity_attached": True,
        "note": "Idle ghost with HIGH security risk"
    },
    {
        "resource_id": "i-6",
        "cpu_avg": 42, "cpu_p95": 61,
        "memory_avg": 55, "network_pct": 38,
        "internet_facing": False, "identity_attached": False,
        "note": "Healthy baseline resource"
    },
    {
        "resource_id": "i-7",
        "cpu_avg": 22, "cpu_p95": 35,
        "memory_avg": 48, "network_pct": 94,
        "internet_facing": True, "identity_attached": False,
        "note": "Network anomaly — possible exfiltration or DDoS"
    },
]


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Accept optional input file path
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        with open(input_path) as f:
            resources = json.load(f)
        print(f"Loaded {len(resources)} resources from {input_path}")
    else:
        resources = TEST_RESOURCES

    results = analyze_batch(resources)

    # Print results
    print(json.dumps(results, indent=2))

    # Save to file
    output_path = "sample_outputs.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to {output_path}")
