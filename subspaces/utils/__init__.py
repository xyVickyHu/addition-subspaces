"""``subspaces.utils`` — library modules (the things that import nothing of their own).

Submodules:
    io           file/log helpers, JSON task loading, checkpoint discovery
    model        HookedTransformer loading + CUDA-memory diagnostics
    data         ICL datasets, splits, prompt formatting, batch processing
    activations  z-results extraction + caching, FV builders
    intervene    NLL + accuracy under residual-stream interventions
    heads        coefficient-matrix helpers, auto thresholds, auto main-heads
    pca          PCA subspace, projection, paper-§4 period/magnitude fitting
    signals      paper-§5 per-demo extracted-signal decomposition (label peaking + aggregation)
"""

from .activations import (
    compute_FVs_dict,
    compute_mean_z_result,
    compute_z_result_per_task,
    compute_z_results_dict,
    load_z_results_dict,
)
from .data import (
    LimitedTaskDataset,
    create_train_val_test_splits,
    find_output_for_input,
    format_input,
    load_tasks_split_indist_ood,
    process_batch_data_individual,
)
from .heads import (
    auto_main_heads,
    auto_threshold,
    extract_heads_above_threshold,
    find_largest_elements,
    get_largest_element_matrix,
    load_latest_matrix,
    recovery_scan,
    select_heads,
    select_main_heads_by_recovery_weak,
)
from .intervene import (
    compute_nll,
    compute_task_accuracy,
    eval_fv_per_example_correctness,
    intervened_generation_with_accuracy,
    intervened_generation_with_nll,
)
from .io import (
    create_log_file,
    get_last_checkpoint,
    get_No_checkpoint,
    load_task_data,
    sort_files_by_epoch,
)
from .model import load_model_no_grad, print_cuda_memory
from .pca import (
    compute_pca_FVs_dict,
    compute_pca_subspace,
    compute_project_vectors,
    compute_subspace_complement,
    fit_mod_vectors,
    opt_shift,
    subspace_principal_angles,
)
from .signals import (
    MAIN_HEADS,
    build_addk_prompt,
    collect_signals,
    collect_signals_discrete,
    extracted_info,
    forward_capture,
    get_hk_directions,
    head_pca_subspace,
    label_relationship,
    signal_cosine,
    summarize_signals,
    value_head_for_query_head,
)
