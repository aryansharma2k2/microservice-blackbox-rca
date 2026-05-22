#!/usr/bin/env python3
"""Add nodeSelector to Online Boutique Deployments for a 2-node k3s VCL cluster.

The split keeps fault-injection targets physically isolated on Machine 2.
When you cpu_hog cartservice (Machine 2), any anomaly on frontend (Machine 1)
is true fault propagation through the application — not hardware contention.

Machine 1 (role=infra-and-upstream):
    frontend, productcatalogservice, shippingservice, redis-cart,
    adservice, loadgenerator

Machine 2 (role=fault-targets):
    cartservice, currencyservice, emailservice, paymentservice,
    recommendationservice, checkoutservice

Rationale for checkoutservice on Machine 2:
    checkoutservice calls paymentservice, emailservice, and cartservice —
    all fault-injection targets. Placing it on Machine 2 ensures its
    back-pressure response to faults is captured on the same physical node,
    making propagation onset times more consistent and reducing cross-node
    network jitter in the metric signal.

Rationale for adservice on Machine 1:
    adservice is off the critical path (frontend does not wait for it).
    Placing it with the upstream services reduces false hardware-contention
    signals during fault-target experiments on Machine 2.

Usage:
    python infra/node-assignment-patch.py                          # in-place
    python infra/node-assignment-patch.py --input in.yaml --output out.yaml
"""

import argparse
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).parent

# Maps deployment name → node role label value
NODE_ASSIGNMENTS: dict[str, str] = {
    # Machine 1 — upstream callers and off-critical-path services
    "frontend":               "infra-and-upstream",
    "productcatalogservice":  "infra-and-upstream",
    "shippingservice":        "infra-and-upstream",
    "redis-cart":             "infra-and-upstream",
    "adservice":              "infra-and-upstream",
    "loadgenerator":          "infra-and-upstream",
    # Machine 2 — fault injection targets (including checkoutservice
    # which is a direct caller of paymentservice, emailservice, cartservice)
    "cartservice":            "fault-targets",
    "currencyservice":        "fault-targets",
    "emailservice":           "fault-targets",
    "paymentservice":         "fault-targets",
    "recommendationservice":  "fault-targets",
    "checkoutservice":        "fault-targets",
}


def patch_deployment(doc: dict) -> str | None:
    name = doc.get("metadata", {}).get("name", "")
    role = NODE_ASSIGNMENTS.get(name)
    if role is None:
        return None
    doc["spec"]["template"]["spec"]["nodeSelector"] = {"role": role}
    return role


def main() -> None:
    default = str(SCRIPT_DIR / "boutique-manifests-patched.yaml")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",  default=default, help="Input manifest path")
    parser.add_argument("--output", default=default, help="Output manifest path (default: in-place)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    docs = [d for d in yaml.safe_load_all(input_path.read_text()) if d is not None]

    patched = 0
    for doc in docs:
        if doc.get("kind") != "Deployment":
            continue
        role = patch_deployment(doc)
        if role:
            name = doc.get("metadata", {}).get("name", "?")
            print(f"[node-assign] deployment/{name:30s} → role={role}")
            patched += 1
        else:
            name = doc.get("metadata", {}).get("name", "?")
            print(f"[node-assign] deployment/{name:30s}   (no assignment — left unscheduled)")

    output_path.write_text(
        yaml.dump_all(docs, default_flow_style=False, allow_unicode=True)
    )
    print(f"[node-assign] {patched} deployment(s) updated → {output_path}")


if __name__ == "__main__":
    main()
