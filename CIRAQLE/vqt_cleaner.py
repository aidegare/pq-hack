"""Variational Quantum Thermalizer (VQT) for the LxL periodic TFIM in Guppy."""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from guppylang import guppy
from guppylang.std.angles import angle
from guppylang.std.builtins import array, comptime, output
from guppylang.std.qsystem.utils import get_current_shot
from guppylang.std.quantum import cx, qubit, rx, ry, rz
from pushquantum.program import current_run, with_measurement

BASES = ("X", "Y", "Z")
SHOTS = 128
TEMPERATURES = (0.01, 1.1875, 2.375, 4.0)


# ---------------------------------------------------------
# 1. Lattice & Classical Thermal Latent Sampling
# ---------------------------------------------------------
def build_edges(L: int) -> list[tuple[int, int]]:
    """Return nearest-neighbor bonds on an LxL periodic torus (q = x + L*y)."""
    edges = []
    for y in range(L):
        for x in range(L):
            q = x + L * y
            edges.append((q, ((x + 1) % L) + L * y))
            edges.append((q, x + L * ((y + 1) % L)))
    return edges


def classical_bond_sums(L: int) -> np.ndarray:
    """Compute sum_<ij> s_i s_j for every bitstring in {0, 1}^(L^2)."""
    N = L * L
    states = np.arange(1 << N, dtype=np.uint32)
    spins = 1.0 - 2.0 * ((states[:, None] >> np.arange(N)) & 1)
    return sum(spins[:, i] * spins[:, j] for i, j in build_edges(L))


def sample_thermal_configs(
    L: int, k: float, n_samples: int, seed: int, use_mc: bool = False
) -> np.ndarray:
    """Sample n_samples spin bitstrings z in {0, 1}^(L^2) from p_k(z) ~ exp(k * sum_<ij> s_i s_j)."""
    N = L * L
    rng = np.random.default_rng(seed)

    # Exact Boltzmann sampling over all 2^N configurations
    if not use_mc:
        bond_sums = classical_bond_sums(L)
        weights = np.exp(k * (bond_sums - bond_sums.max()))
        chosen = rng.choice(1 << N, size=n_samples, p=weights / weights.sum())
        return ((chosen[:, None] >> np.arange(N)) & 1).astype(np.int64)

    # Metropolis Monte Carlo sampling when use_mc=True
    spins = rng.choice([-1, 1], size=(L, L))
    configs = np.empty((n_samples, N), dtype=np.int64)
    burn_in, stride = 200 * N, 10 * N

    for step in range(burn_in + n_samples * stride):
        x, y = int(rng.integers(L)), int(rng.integers(L))
        s = spins[y, x]
        neighbors = (
            spins[(y + 1) % L, x] + spins[(y - 1) % L, x]
            + spins[y, (x + 1) % L] + spins[y, (x - 1) % L]
        )
        dE = 2.0 * s * neighbors
        if dE <= 0.0 or rng.random() < math.exp(-k * dE):
            spins[y, x] = -s

        if step >= burn_in and (step - burn_in) % stride == 0:
            # Flip all spins with 50% probability to enforce global Z2 symmetry
            oriented = -spins if rng.random() < 0.5 else spins
            configs[(step - burn_in) // stride] = (1 - oriented.reshape(-1)) // 2

    return configs


# ---------------------------------------------------------
# 2. Variational Free-Energy Optimization
# ---------------------------------------------------------
def optimize_vqt(temperature: float, L: int = 4, J: float = 1.0, h: float = 1.0) -> dict:
    """Minimize Helmholtz free energy F = <H> - T * S(rho) over (k, theta, gamma, alpha)."""
    L_opt = min(L, 4)
    N = L_opt * L_opt
    edges = build_edges(L_opt)

    # Precompute neighbor index tables for vectorized Pauli expectation values
    nbrs = [[] for _ in range(N)]
    for i, j in edges:
        nbrs[i].append(j)
        nbrs[j].append(i)
    nbr_arr = np.array(nbrs, dtype=np.int64)
    edge_i = np.array([i for i, _ in edges], dtype=np.int64)
    edge_j = np.array([j for _, j in edges], dtype=np.int64)
    nbr_i_no_j = np.array([[v for v in nbrs[i] if v != j] for i, j in edges], dtype=np.int64)
    nbr_j_no_i = np.array([[v for v in nbrs[j] if v != i] for i, j in edges], dtype=np.int64)

    # Sort classical states by bond energy for deterministic quantile sampling
    bond_sums = classical_bond_sums(L_opt)
    order = np.argsort(bond_sums)
    sorted_bonds = bond_sums[order]
    quantiles = (np.arange(64) + 0.5) / 64
    k_cache: dict[float, tuple[float, np.ndarray]] = {}

    def get_k_data(k: float) -> tuple[float, np.ndarray]:
        key = round(k, 6)
        if key not in k_cache:
            weights = np.exp(key * (sorted_bonds - sorted_bonds[-1]))
            probs = weights / weights.sum()
            s_classical = float(-np.sum(probs * np.log(probs + 1e-15)))
            chosen = order[np.searchsorted(np.cumsum(probs), quantiles)]
            spins = 1.0 - 2.0 * (((chosen[:, None] >> np.arange(N)) & 1).astype(np.float64))
            k_cache[key] = (s_classical, spins)
        return k_cache[key]

    def free_energy(k: float, theta: float, gamma: float, alpha: float) -> float:
        s_classical, spins = get_k_data(k)

        # Scale classical entropy by binary entropy of non-orthogonal tilted states (<psi_0|psi_1> = sin(theta))
        p_plus = 0.5 * (1.0 + math.sin(theta))
        if p_plus <= 1e-12 or p_plus >= 1.0 - 1e-12:
            entropy = 0.0
        else:
            h2 = -p_plus * math.log(p_plus) - (1.0 - p_plus) * math.log(1.0 - p_plus)
            entropy = s_classical * (h2 / math.log(2.0))

        # Exact Heisenberg-picture <sum X_q> and <sum Z_i Z_j> for RY(theta) -> RZZ(gamma) -> RX(alpha)
        ct, st = math.cos(theta), math.sin(theta)
        cg, sg = math.cos(gamma), math.sin(gamma)
        ca, sa = math.cos(alpha), math.sin(alpha)

        mz = spins * ct
        phase = cg + 1j * sg * mz

        prod_q = np.prod(phase[:, nbr_arr], axis=2)
        exp_x = st * float(np.mean(np.sum(prod_q.real, axis=1)))

        prod_i = np.prod(phase[:, nbr_i_no_j], axis=2)
        prod_j = np.prod(phase[:, nbr_j_no_i], axis=2)
        mzi, mzj = mz[:, edge_i], mz[:, edge_j]

        zz = (ca ** 2) * (mzi * mzj)
        zy = st * (sg * prod_j.real + cg * mzi * prod_j.imag)
        yz = st * (sg * prod_i.real + cg * mzj * prod_i.imag)
        yy = (st ** 2) * (prod_i.imag * prod_j.imag)
        exp_zz = float(np.mean(np.sum(zz + sa * ca * (zy + yz) + (sa ** 2) * yy, axis=1)))

        return (-J * exp_zz - h * exp_x) - temperature * entropy

    # Coarse grid initialization over (k, theta)
    best = [0.5, 0.3, 0.0, 0.0]
    best_F = float("inf")
    for k in np.linspace(0.0, 3.0, 25):
        for th in np.linspace(0.0, 0.5 * math.pi, 25):
            val = free_energy(float(k), float(th), 0.0, 0.0)
            if val < best_F:
                best_F, best = val, [float(k), float(th), 0.0, 0.0]

    # Coordinate and diagonal pattern search over (k, theta, gamma, alpha)
    bounds = [
        (0.0, 5.0),
        (0.0, 0.5 * math.pi),
        (-0.5 * math.pi, 0.5 * math.pi),
        (-0.5 * math.pi, 0.5 * math.pi),
    ]
    steps = [0.15, 0.08, 0.08, 0.08]
    directions = [
        (1, 0, 0, 0), (-1, 0, 0, 0),
        (0, 1, 0, 0), (0, -1, 0, 0),
        (0, 0, 1, 0), (0, 0, -1, 0),
        (0, 0, 0, 1), (0, 0, 0, -1),
        (0, 0, 1, 1), (0, 0, -1, -1),
        (0, 0, 1, -1), (0, 0, -1, 1),
    ]

    for _ in range(30):
        improved = False
        for d in directions:
            cand = [float(np.clip(best[i] + d[i] * steps[i], *bounds[i])) for i in range(4)]
            val = free_energy(*cand)
            if val < best_F - 1e-10:
                best_F, best, improved = val, cand, True
        if not improved:
            steps = [s * 0.5 for s in steps]

    return {
        "k": best[0],
        "theta": best[1],
        "gamma": best[2],
        "alpha": best[3],
        "free_energy": best_F,
    }


# ---------------------------------------------------------
# 3. Guppy Circuit & Shot Multiplexing
# ---------------------------------------------------------
def pack_128_masks(configs: np.ndarray) -> list[list[int]]:
    """Pack 128 shot bits per qubit into two signed 64-bit integer masks [low, high]."""
    masks = []
    for q in range(configs.shape[1]):
        pair = []
        for half in range(2):
            mask = sum(int(configs[64 * half + i, q]) << i for i in range(64))
            pair.append(mask if mask < (1 << 63) else mask - (1 << 64))
        masks.append(pair)
    return masks


@guppy
def tilt_qubit(
    q: qubit, shot: int, low: int, high: int, zero_angle: angle, one_angle: angle
) -> int:
    """Unpack this shot's bit from [low, high] and apply RY(theta) or RY(pi - theta)."""
    bit = ((low >> shot) & 1) if shot < 64 else ((high >> (shot - 64)) & 1)
    ry(q, one_angle if bit == 1 else zero_angle)
    return bit


@guppy
def entangle_zz(q0: qubit, q1: qubit, gamma: angle) -> None:
    """Apply two-qubit interaction exp(-i * gamma/2 * Z0 Z1)."""
    cx(q0, q1)
    rz(q1, gamma)
    cx(q0, q1)


def build_preparation(
    temperature: float = 2.375,
    basis: str = "Z",
    L: int = 4,
    J: float = 1.0,
    h: float = 1.0,
    seed: int = 1821,
    shot_offset: int = 1,
    params: dict | None = None,
    use_mc: bool = False,
):
    """Build the 128-shot Guppy state-preparation program for a given temperature and basis."""
    if params is None:
        params = optimize_vqt(temperature=temperature, L=L, J=J, h=h)

    temp_idx = TEMPERATURES.index(temperature) if temperature in TEMPERATURES else int(temperature * 100)
    basis_seed = seed + temp_idx * 3 + BASES.index(basis)
    configs = sample_thermal_configs(
        L=L, k=params["k"], n_samples=SHOTS, seed=basis_seed, use_mc=use_mc
    )
    masks = pack_128_masks(configs)
    edges = build_edges(L)

    theta, gamma, alpha = params["theta"], params["gamma"], params["alpha"]
    n_qubits = L * L
    N = comptime(n_qubits)

    @guppy.comptime
    def run_vqt_circuit(qs: array[qubit, N]) -> int:
        shot = get_current_shot() - shot_offset
        label = 0

        # Layer 1: Asymmetric single-qubit RY tilts conditioned on classical latent bits
        for q in range(n_qubits):
            low, high = masks[q]
            bit = tilt_qubit(
                qs[q],
                shot,
                low,
                high,
                angle(theta / math.pi),
                angle((math.pi - theta) / math.pi),
            )
            label |= bit << q

        # Layer 2: Nearest-neighbor RZZ entanglement across periodic bonds
        if abs(gamma) > 1e-12:
            for i, j in edges:
                entangle_zz(qs[i], qs[j], angle(gamma / math.pi))

        # Layer 3: Transverse-field RX rotation on every qubit
        if abs(alpha) > 1e-12:
            for q in range(n_qubits):
                rx(qs[q], angle(alpha / math.pi))

        return label

    @guppy
    def main() -> array[qubit, N]:
        qs = array(qubit() for _ in range(N))
        label = run_vqt_circuit(qs)
        output("shot", get_current_shot() - comptime(shot_offset))
        output("label", label)
        return qs

    return main, params


# ---------------------------------------------------------
# 4. Selene Execution & JSON Submission Export
# ---------------------------------------------------------
def run_local_submission(
    temperature: float = 2.375,
    L: int = 4,
    J: float = 1.0,
    h: float = 1.0,
    seed: int = 1821,
    output_path: Path | None = None,
    use_mc: bool = False,
    noisy: bool = False,
    p_1q: float = 1e-5,
    p_2q: float = 1.4e-4,
    p_meas: float = 1e-3,
    p_init: float = 1e-5,
):
    """Optimize VQT parameters once, simulate 128 shots per basis on Selene, and save JSON."""
    from selene_sim import DepolarizingErrorModel, Quest, build

    params = optimize_vqt(temperature=temperature, L=L, J=J, h=h)
    error_model = (
        DepolarizingErrorModel(p_1q=p_1q, p_2q=p_2q, p_meas=p_meas, p_init=p_init)
        if noisy
        else None
    )

    submission = {"temperature": float(temperature)}
    n_qubits = L * L
    edges = build_edges(L)

    for basis in BASES:
        prepare, _ = build_preparation(
            temperature=temperature,
            basis=basis,
            L=L,
            J=J,
            h=h,
            seed=seed,
            shot_offset=0,
            params=params,
            use_mc=use_mc,
        )
        runner = build(with_measurement(prepare, basis.lower()).compile())
        shot_iter = (
            runner.run_shots(Quest(), n_qubits=n_qubits, n_shots=SHOTS, error_model=error_model)
            if error_model is not None
            else runner.run_shots(Quest(), n_qubits=n_qubits, n_shots=SHOTS)
        )

        rows = [[int(b) for b in dict(shot)["bits"]] for shot in shot_iter]
        submission[basis] = rows

        spins = 1.0 - 2.0 * np.array(rows)
        msg = f"T={temperature:g} | Basis {basis}: <{basis}> = {spins.mean():+.4f}"
        if basis == "Z":
            zz = np.mean([spins[:, i] * spins[:, j] for i, j in edges])
            msg += f" | <Z_i Z_j> = {zz:+.4f}"
        print(msg)

    out_file = output_path or Path(f"submission_{temperature:g}{'_noisy' if noisy else ''}.json")
    with out_file.open("w") as f:
        f.write('{\n  "temperature": ' + json.dumps(float(temperature)) + ",\n")
        for basis in BASES:
            f.write(f'  "{basis}": [\n')
            f.write(",\n".join("    " + json.dumps(row) for row in submission[basis]))
            f.write("\n  ]" + ("," if basis != "Z" else "") + "\n")
        f.write("}\n")

    print(
        f"Saved {out_file} | k={params['k']:.3f}, theta={params['theta']:.3f}, "
        f"gamma={params['gamma']:.3f}, alpha={params['alpha']:.3f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperature", type=float, default=2.375)
    parser.add_argument("--L", type=int, default=4)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1821)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--mc", action="store_true", help="Use Metropolis Monte Carlo sampling")
    parser.add_argument("--noisy", action="store_true", help="Enable Selene DepolarizingErrorModel")
    parser.add_argument("--p-1q", type=float, default=1e-5)
    parser.add_argument("--p-2q", type=float, default=1.4e-4)
    parser.add_argument("--p-meas", type=float, default=1e-3)
    parser.add_argument("--p-init", type=float, default=1e-5)
    args = parser.parse_args()

    run_local_submission(
        temperature=args.temperature,
        L=args.L,
        J=args.J,
        h=args.h,
        seed=args.seed,
        output_path=args.output,
        use_mc=args.mc,
        noisy=args.noisy,
        p_1q=args.p_1q,
        p_2q=args.p_2q,
        p_meas=args.p_meas,
        p_init=args.p_init,
    )
elif (run := current_run.get()) is not None:
    if run.shots != SHOTS:
        raise SystemExit(f"VQT requires exactly {SHOTS} shots.")
    temp = float(getattr(run, "temperature", None) or os.environ.get("VQT_TEMPERATURE", "2.375"))
    use_mc = bool(getattr(run, "use_mc", False))
    main, _ = build_preparation(
        temperature=temp, basis=run.basis.upper(), shot_offset=run.shot_offset, use_mc=use_mc
    )