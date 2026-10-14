"""Readable offline request cards and build summaries; no simulated predictions."""

import html
import json

import pandas as pd

from src.jev.provider import api_preview
from src.utils.notebook_report import record


def show_request(request):
    from IPython.display import HTML, display

    body = api_preview(request)
    origin = json.loads(request.provenance_json)
    task = body["state"]["task"]
    question = body["questions"]["semantic"]
    escape = html.escape
    text = "".join(
        f"<h4>{escape(column)} · {len(str(value)):,} characters</h4>"
        f"<blockquote>{escape(str(value)[:650])}{' … [display excerpt]' if len(str(value)) > 650 else ''}</blockquote>"
        for column, value in body["state"]["text"].items()
    )
    feature = {
        "noul": "One numeric feature: P(positive class), between 0 and 1.",
        "choice": "K numeric features: one probability for each target class. They sum to 1.",
        "score": "One main numeric feature: Σ direction × P(direction), between −4 and +4. Retain all nine probabilities.",
    }[question["type"]]
    display(
        HTML(
            f"<h3>{escape(origin['dataset_id'])} · {escape(origin['representation'])}</h3>"
            f"<p><b>Source row:</b> {escape(str(origin['row_id']))} · <b>Target name:</b> {escape(str(task['target']))}</p>"
            + text
            + f"<p><b>Fixed question:</b> {escape(question['instructions'])}</p>"
            + f"<p><b>Output:</b> {feature} This preview performs no inference.</p>"
            + "<details><summary>Complete API body, including answer options</summary><pre>"
            + escape(json.dumps(body, ensure_ascii=False, indent=2))
            + "</pre></details>"
        )
    )
    record(
        {"origin": origin, "api_body": body, "expected_feature_schema": feature},
        "Actual request example — no API call",
    )


def scope_table(audit, plan):
    widths = plan.pivot(index="dataset_id", columns="mode", values="main_numeric_columns")
    table = audit.summary.set_index("dataset_id").join(widths, how="inner")
    table["main_features_both"] = (table.per_column + table.joint).where(
        table.text_columns.ne(1), table.per_column
    )
    return table[
        [
            "dataset",
            "task_type",
            "rows",
            "text_columns",
            "per_column",
            "joint",
            "main_features_both",
            "combined_unique_inputs",
        ]
    ].reset_index(drop=True)


def dataset_budget(estimates, ratio=4):
    frame = estimates[estimates.characters_per_token.eq(ratio)]
    result = (
        frame.groupby("dataset", sort=False)[
            ["requests_for_combined_build", "wire_tokens_combined_approx"]
        ]
        .sum()
        .reset_index()
    )
    return result


def mode_budget(estimates, cfg, ratio=4):
    frame = estimates[estimates.characters_per_token.eq(ratio)]
    records = []
    for mode in ("per_column", "joint", "both"):
        group = frame if mode == "both" else frame[frame["mode"].eq(mode)]
        requests = (
            group.requests_for_combined_build.sum()
            if mode == "both"
            else group.unique_requests_within_group.sum()
        )
        tokens = (
            group.wire_tokens_combined_approx.sum()
            if mode == "both"
            else group.wire_tokens_unique_approx.sum()
        )
        records.append(
            {
                "mode": mode,
                "unique_requests": int(requests),
                "input_tokens_approx": int(tokens),
                "cost_usd": None
                if cfg["cost"]["input_price_per_million"] is None
                else tokens * cfg["cost"]["input_price_per_million"] / 1e6,
            }
        )
    return pd.DataFrame(records)


def throughput_table(requests, tokens, cfg):
    tps = cfg["throughput_review"]["proposed_input_tokens_per_second"]
    proposed_rps = cfg["throughput_review"]["proposed_requests_per_second"]
    return pd.DataFrame(
        [
            {
                "requests_per_second": rps,
                "input_tokens_per_second": tps,
                "request_bound_hours": requests / rps / 3600,
                "token_bound_hours": tokens / tps / 3600,
                "ideal_minimum_hours": max(requests / rps, tokens / tps) / 3600,
            }
            for rps in (proposed_rps / 2, proposed_rps, proposed_rps * 2)
        ]
    )
