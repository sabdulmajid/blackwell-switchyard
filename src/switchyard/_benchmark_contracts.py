"""Immutable source and work contracts for the Liger benchmark comparators."""

PINNED_LIGER_COMMIT = "777799588a89d74c489ed995e3bf006427738e85"
PINNED_LIGER_SOURCE_SHA256 = (
    "57da6fed98f794088b2a56223e6c7ef9fc920824f0c483cb0ef0b5a343dab0b1"
)

LIGER_EXACT_WORK_CONTRACT = {
    "contract_version": 1,
    "functional_inputs": ["v", "w"],
    "differentiated_inputs": ["v", "w"],
    "auxiliary_inputs": [],
    "returned_gradients": ["dv", "dw"],
    "discarded_gradients": [],
    "forward_source_passes": 2,
    "backward_source_passes": 2,
    "saved_fp32_scalars_per_source_token": 2,
    "gradient_atomic_vectors_per_token": 1,
    "atomic_gradient_roles": ["dw"],
    "extra_work_disclosed": False,
    "role": "promotion comparator",
}
LIGER_UPSTREAM_WORK_CONTRACT = {
    **LIGER_EXACT_WORK_CONTRACT,
    "auxiliary_inputs": ["gain"],
    "discarded_gradients": ["d_gain"],
    "gradient_atomic_vectors_per_token": 2,
    "atomic_gradient_roles": ["dw", "discarded_d_gain"],
    "extra_work_disclosed": True,
    "role": "observational upstream comparator",
}
