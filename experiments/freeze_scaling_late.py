"""Secondary prospective forecasts from measured late LLM gradient geometry."""

import json
from pathlib import Path
import time

import numpy as np

from analyze_scaling_digit import power_prediction
from run_elicitation_pilot import ROOT, digest, save
from train_bestn import stamp


def main():
    output=ROOT/"results/scaling_s4g/late_forecast";output.mkdir(exist_ok=False)
    source=Path(__file__);(output/source.name).write_bytes(source.read_bytes())
    save(output/"protocol.json",{"created_at_uk":stamp(),"source_sha256":digest(source),"seeds":[99201,99202,99203],
         "status":"Secondary prospective test added after two confirmation trajectories reached1024 and rejected the early cubic extrapolation. The original primary forecast remains retained.",
         "prediction":"Use each run's exact update1024 error and current training/evaluation gradient overlap, with its already fixed learning rate. Primary late hypothesis is constant quadratic gradient/error geometry, the separable Boltzmann-gate limit. Gamma3 and4 remain explicit sensitivity alternatives. Assume error floor0; no exponent, coefficient or time shift is fitted to subsequent observations.",
         "targets":[1536,2048],"knowledge":"For each numerical forecast, record the last already visible checkpoint. Only checkpoints beyond that cutoff count as prospective. A twofold late window can test a rate forecast, not establish an asymptotic scaling law."})
    pending={99201,99202,99203}
    while pending:
        for seed in sorted(pending):
            root=ROOT/"results/scaling_s4g/digit/confirmation"/str(seed)/"current"
            paths=list(root.glob("attempt*/gradient_1024.json"))
            if not paths:continue
            gradient=json.loads(max(paths).read_text())
            history=json.loads((root/"progress.json").read_text())
            error=next(r["error"] for r in history["evaluations"] if r["step"]==1024)
            forecasts={str(gamma):[{"step":step,"error":float(value)} for step,value in zip([1536,2048],power_prediction(1024,error,gradient["current_success_gradient_overlap"],history["eta"],gamma,np.array([1536,2048])),strict=True)] for gamma in (2,3,4)}
            record={"written_at_uk":stamp(),"seed":seed,"anchor_step":1024,"anchor_error":error,"gradient_overlap":gradient["current_success_gradient_overlap"],
                    "eta":history["eta"],"already_observed_through":history["step"],"predictions":forecasts}
            save(output/f"{seed}.json",record);print(json.dumps(record),flush=True);pending.remove(seed)
        if pending:time.sleep(10)
    save(output/"complete.json",{"finished_at_uk":stamp(),"seeds":[99201,99202,99203]})


if __name__=="__main__":main()
