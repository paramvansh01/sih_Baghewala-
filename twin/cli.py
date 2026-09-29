"""Command-line entry point.

  python -m twin.cli params                 list every parameter with provenance
  python -m twin.cli generate               synthesise history + card datasets
  python -m twin.cli train                  train all models from scratch
  python -m twin.cli optimise               CSS optimiser + SRP policy evaluation (physics-verified)
  python -m twin.cli report                 write artifacts/report.md + figures
  python -m twin.cli all                    everything above, in order
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)


def cmd_params(_):
    import pandas as pd

    from .config import FieldConfig
    df = pd.DataFrame(FieldConfig().provenance_table())
    pd.set_option("display.max_colwidth", 90, "display.width", 250)
    print(df.to_string(index=False))
    print("\ncounts:", df.status.value_counts().to_dict())


def cmd_generate(_):
    from .config import FieldConfig, load_pipeline
    from .data import generate_cards, generate_history
    t = time.time()
    w, c, d = generate_history.generate()
    rate = generate_history.field_rate_bopd(d, FieldConfig().value("field.history_end_date"))
    print(f"history: {len(w)} wells, {len(c)} cycles, {len(d)} daily rows; "
          f"synthetic field rate (last 365 d) {rate:.0f} bopd  [{time.time() - t:.1f}s]")
    t = time.time()
    m, _, f = generate_cards.generate()
    print(f"cards: {len(m)} ({f} failed physics checks)  [{time.time() - t:.1f}s]")
    pl = load_pipeline()
    r = pl["card_cnn"]["robustness"]
    m, _, f = generate_cards.generate(n_per_class=r["n_per_class"], seed=pl["seed"] + r["seed_offset"],
                                      noise_mult=r["noise_multiplier"], name="cards_robust")
    print(f"robustness cards: {len(m)} ({f} failed)")


def cmd_train(_):
    from .models import card_cnn, css_surrogate, trajectory
    t = time.time()
    _, res, info = css_surrogate.train()
    print(css_surrogate.summary_table(res).round(3).to_string(index=False))
    print(f"[surrogate {time.time() - t:.1f}s]")
    t = time.time()
    _, res, _ = trajectory.train()
    print({k: {m: round(v["r2"], 3) for m, v in r.items() if isinstance(v, dict)} for k, r in res.items()},
          f"[trajectory {time.time() - t:.1f}s]")
    t = time.time()
    _, r = card_cnn.train(verbose=False)
    print(f"card CNN: test acc {r['cnn']['accuracy']:.4f}, macro-F1 {r['cnn']['macro_f1']:.4f}, "
          f"robustness acc {r.get('robustness', {}).get('accuracy', float('nan')):.4f}  [{time.time() - t:.1f}s]")


def cmd_optimise(_):
    from .optimize import css_optimizer, srp_eval
    from .config import ROOT
    out = ROOT / "artifacts" / "metrics"
    t = time.time()
    res = css_optimizer.run()
    with open(out / "css_optimisation.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    s = css_optimizer.summarise(res)
    s.to_csv(out / "css_optimisation_summary.csv", index=False)
    print(s.describe().round(2).to_string(), f"[css optimiser {time.time() - t:.1f}s]")
    t = time.time()
    srp = srp_eval.run()
    with open(out / "srp_evaluation.json", "w") as f:
        json.dump(srp, f, indent=2, default=float)
    print(json.dumps(srp["field_totals"], indent=1, default=float), f"[srp eval {time.time() - t:.1f}s]")


def cmd_livecheck(_):
    from .config import ROOT
    from .live.engine import validate_live_cards
    t = time.time()
    r = validate_live_cards()
    with open(ROOT / "artifacts" / "metrics" / "live_card_check.json", "w") as f:
        json.dump(r, f, indent=2)
    print(f"live cards: exact {r['exact_agreement']:.3f}, grouped {r['grouped_agreement']:.3f} "
          f"(n={r['n_live_cards']}); injected faults {r['fault_agreement']:.3f}  [{time.time() - t:.1f}s]")


def cmd_report(_):
    from . import report
    print("wrote", report.build())


def cmd_serve(a):
    from . import serve
    serve.serve(a.port)


def cmd_live(a):
    from .live import server
    server.run(a.port)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("commands", nargs="+",
                    choices=["params", "generate", "train", "optimise", "livecheck", "report", "serve", "live",
                             "all"])
    ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args()
    for cmd in a.commands:
        if cmd == "all":
            for c in (cmd_generate, cmd_train, cmd_optimise, cmd_livecheck, cmd_report):
                c(a)
        else:
            globals()[f"cmd_{cmd}"](a)


if __name__ == "__main__":
    main()
