"""Independent long-run confirmation of the predeclared finite-SGD laws."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from run_elicitation_pilot import ROOT, digest, save, setup
from scaling_theory import asymptotic_constant, phi
from simulate_scaling_sgd import simulate
from train_bestn import stamp


def main():
    output = ROOT/"results/scaling_s4g/finite_confirmation"
    raw = setup()/"scaling_s4g/finite_confirmation"
    output.mkdir(parents=True, exist_ok=False)
    raw.mkdir(parents=True, exist_ok=False)
    (output/"source").mkdir()
    names = ["confirm_scaling_sgd.py", "simulate_scaling_sgd.py", "scaling_theory.py"]
    for name in names: (output/"source"/name).write_bytes((ROOT/"experiments"/name).read_bytes())
    maximum, replicas = 262144, 512
    checks = sorted({0, *np.unique(np.rint(np.geomspace(1, maximum, 97)).astype(int)).tolist()})
    cases = [(1,8,.1), (8,1,.2), (8,2,.2)]
    specifications = [{"index": i, "dimension": d, "batch": b, "eta": eta, "source": "current", "replicas": replicas,
                       "steps": maximum, "checks": checks, "seed": 97000+i, "output": str(output), "raw": str(raw)} for i,(d,b,eta) in enumerate(cases)]
    predictions = []
    for d,b,eta in cases:
        lam = (d-1)*eta*float(phi(1.5))/b
        beta = .5 if d==1 else 1/3
        coefficient = np.sqrt(asymptotic_constant()/8) if d==1 else lam**(2/3)/(2*3**(1/3))
        predictions.append({"dimension":d,"batch":b,"eta":eta,"beta":beta,"coefficient":float(coefficient)})
    save(output/"protocol.json", {"created_at_uk":stamp(), "source_sha256":{name:digest(output/"source"/name) for name in names},
         "specifications":specifications,"predictions":predictions,"fit_window_steps":[maximum//8,maximum],
         "status":"Fresh RNG seeds and twice as many replicas, four times the pilot horizon. Parameters and asymptotic coefficients fixed before these samples. The second noisy batch size tests the predicted eta/batch dependence without refitting.",
         "assessment":"Whole-replica bootstrap slopes over the fixed final factor8; coefficient ratios to unfitted predictions; report finite-range deviations from asymptotic exponents. The existing stationarity rule still applies."})
    with ProcessPoolExecutor(max_workers=3) as pool:
        results = list(tqdm(pool.map(simulate,specifications),total=len(specifications),desc="Fresh finite-SGD confirmation"))
    save(output/"complete.json",results)


if __name__ == "__main__": main()
