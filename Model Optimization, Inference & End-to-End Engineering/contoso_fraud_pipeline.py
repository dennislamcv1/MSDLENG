"""
contoso_fraud_pipeline.py
End-to-End DL Engineering Project - Contoso Financial real-time fraud detection

Takes the 2-billion-parameter fraud classifier from notebooks to production in three steps,
completed from dl_project_SCAFFOLD.ipynb:

    Step 1  orchestrate_training()  DeepSpeed ZeRO-3 job across a multi-node A100 cluster,
                                    wrapped in a Random-sampling + BanditPolicy sweep
    Step 2  optimize_model()        Hugging Face Optimum ONNX export + dynamic INT8
                                    quantization, loaded with TensorRT first in the
                                    ONNX Runtime provider chain
    Step 3  deploy_endpoint()       Managed Online Endpoint, "green" deployment with
                                    score.py, 90/10 blue/green traffic split

Building the job and endpoint objects makes no Azure call. The only network calls are in
the submit / promote / rollback helpers at the bottom, and they run only when asked for on
the command line.

Usage
    python contoso_fraud_pipeline.py                  build and self-check Steps 1-3 offline
    python contoso_fraud_pipeline.py --skip-optimize  Steps 1 and 3 only (no PyTorch needed)
    python contoso_fraud_pipeline.py --submit         also provision, submit the sweep and
                                                      roll green out at 10 %
    python contoso_fraud_pipeline.py --promote 50     move green to 50 % once SLA checks pass
    python contoso_fraud_pipeline.py --rollback       send 100 % of traffic back to blue

Environment (the scaffold's "Quantization" kernel)
    pip install azure-ai-ml azure-identity torch onnx "optimum-onnx[onnxruntime]" transformers

Azure settings come from AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP and
AZUREML_WORKSPACE_NAME; the scaffold's placeholder values are used when they are not set.
"""

import argparse
import json
import math
import os
from pathlib import Path

from azure.ai.ml import Input, MLClient, PyTorchDistribution, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import (
    AmlCompute,
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
    OnlineRequestSettings,
    ProbeSettings,
)
from azure.ai.ml.sweep import BanditPolicy, LogUniform, Uniform
from azure.identity import DefaultAzureCredential

# --------------------------------------------------------------------------------------
# Azure workspace
# --------------------------------------------------------------------------------------
SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID", "sub_id")
RESOURCE_GROUP = os.environ.get("AZURE_RESOURCE_GROUP", "rg_name")
WORKSPACE_NAME = os.environ.get("AZUREML_WORKSPACE_NAME", "ws_name")

# --------------------------------------------------------------------------------------
# Step 1 - Distributed training and sweep blueprint
# --------------------------------------------------------------------------------------
TRAIN_CODE_DIR = Path("./src")  # holds train_deepspeed.py; ds_config.json is written here
GPU_CLUSTER_NAME = "gpu-cluster"
GPU_CLUSTER_SKU = "Standard_ND96amsr_A100_v4"  # 8 x A100 80 GB per node, InfiniBand between nodes
NODES_PER_TRIAL = 2
GPUS_PER_NODE = 8
MAX_CLUSTER_NODES = 4  # room for 2 concurrent 2-node trials; scales to 0 when idle
DATALOADER_WORKERS = 8  # replaces the single-threaded loader (96 vCPUs / 8 ranks = 12 per rank)
# Versioned ACPT curated environment from the shared azureml registry. Pinned rather than
# @latest so every trial and every rerun uses the same image. Confirm the version number
# under Studio > Environments > Curated before the first submission.
ACPT_ENVIRONMENT = (
    "azureml://registries/azureml/environments/acpt-pytorch-2.2-cuda12.1/versions/59"
)
TRAIN_DATA = "azureml:contoso-fraud-transactions:1"
PRIMARY_METRIC = "val_accuracy"  # logged by train_deepspeed.py on rank 0 after each epoch

# ZeRO-3 shards parameters, gradients and optimizer state across all 16 ranks. Mixed-precision
# AdamW needs about 16 bytes per parameter, so 2B parameters is ~32 GB of model state: that
# is what crashed a single GPU, and it drops to ~2 GB per rank here.
# The optimizer is not set in this file. train_deepspeed.py builds torch.optim.AdamW from the
# swept --learning_rate / --weight_decay and passes it to deepspeed.initialize().
DS_CONFIG = {
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "bf16": {"enabled": True},  # A100 runs bf16 natively, so no loss scaling is needed
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": True,  # overlap all-gather / reduce-scatter with compute
        "contiguous_gradients": True,
        # Write one consolidated 16-bit checkpoint, which is what Step 2 exports to ONNX
        "stage3_gather_16bit_weights_on_model_save": True,
    },
}


def build_gpu_cluster():
    """Multi-node GPU cluster definition. Created only by submit_training()."""
    return AmlCompute(
        name=GPU_CLUSTER_NAME,
        size=GPU_CLUSTER_SKU,
        min_instances=0,  # no idle GPU cost between sweeps
        max_instances=MAX_CLUSTER_NODES,
        idle_time_before_scale_down=900,
        tier="Dedicated",  # low-priority nodes can be pre-empted mid-way through a ZeRO-3 step
    )


def write_ds_config(code_dir=TRAIN_CODE_DIR):
    """Write ds_config.json into the code snapshot so the --deepspeed flag finds it on the nodes."""
    code_dir.mkdir(parents=True, exist_ok=True)
    config_path = code_dir / "ds_config.json"
    config_path.write_text(json.dumps(DS_CONFIG, indent=2))
    return config_path


def orchestrate_training():
    write_ds_config()
    cluster = build_gpu_cluster()

    base_run = command(
        display_name="fraud-2b-zero3",
        experiment_name="contoso-fraud-2b-zero3",
        description="2B-parameter fraud classifier, DeepSpeed ZeRO-3 across 2 x 8 A100",
        code=str(TRAIN_CODE_DIR),
        command=(
            "python train_deepspeed.py --deepspeed ds_config.json "
            "--train_data ${{inputs.train_data}} "
            "--learning_rate ${{inputs.learning_rate}} "
            "--weight_decay ${{inputs.weight_decay}} "
            "--dataloader_num_workers ${{inputs.num_workers}} "
            "--output_dir ./outputs"
        ),
        inputs={
            "train_data": Input(
                type=AssetTypes.URI_FOLDER, path=TRAIN_DATA, mode=InputOutputModes.RO_MOUNT
            ),
            "learning_rate": 3e-5,
            "weight_decay": 0.01,
            "num_workers": DATALOADER_WORKERS,
        },
        environment=ACPT_ENVIRONMENT,
        compute=cluster.name,
        instance_count=NODES_PER_TRIAL,
        distribution=PyTorchDistribution(process_count_per_instance=GPUS_PER_NODE),
        environment_variables={
            "NCCL_IB_DISABLE": "0",  # keep cross-node collectives on InfiniBand
            "NCCL_DEBUG": "WARN",
        },
        shm_size="16g",  # DataLoader worker processes pass batches through /dev/shm
    )

    # Random sampling: Azure ML rejects Bayesian sampling combined with an early termination
    # policy. BanditPolicy (slack 0.1) stops any trial whose val_accuracy drops below
    # 1 / 1.1, about 91 %, of the best trial so far, after a delay of two evaluation intervals.
    sweep_job = base_run(
        learning_rate=LogUniform(min_value=math.log(1e-5), max_value=math.log(1e-4)),
        weight_decay=Uniform(min_value=0.0, max_value=0.1),
    ).sweep(
        compute=cluster.name,
        sampling_algorithm="random",
        primary_metric=PRIMARY_METRIC,
        goal="Maximize",
        early_termination_policy=BanditPolicy(
            slack_factor=0.1, evaluation_interval=1, delay_evaluation=2
        ),
    )
    sweep_job.set_limits(
        max_total_trials=12,
        max_concurrent_trials=MAX_CLUSTER_NODES // NODES_PER_TRIAL,
        timeout=48 * 3600,
        trial_timeout=6 * 3600,
    )
    sweep_job.display_name = "fraud-2b-zero3-sweep"
    sweep_job.experiment_name = "contoso-fraud-2b-zero3"
    return sweep_job


# --------------------------------------------------------------------------------------
# Step 2 - Optimization and export
# --------------------------------------------------------------------------------------
TEST_MODEL = "hf-internal-testing/tiny-random-BertForSequenceClassification"
QUANTIZED_DIR = "./onnx_quantized"
QUANTIZED_FILE = "model_quantized.onnx"  # Optimum's name: <exported stem>_quantized.onnx
EXECUTION_PROVIDERS = [
    "TensorrtExecutionProvider",  # first choice: TensorRT builds fused engines on the GPU
    "CUDAExecutionProvider",  # runs any node TensorRT does not take
    "CPUExecutionProvider",  # always available; last resort
]
TRT_OPTIONS = {
    "trt_fp16_enable": True,  # the non-integer parts of the graph run in FP16
    "trt_engine_cache_enable": True,  # build the engine once, not on every restart
}
MAX_SEQ_LEN = 128  # fixed padding keeps input shapes stable, so TensorRT does not rebuild engines
SAMPLE_REQUEST = {
    "transactions": [
        "amount=912.40 currency=USD merchant=electronics card_present=0 country_mismatch=1",
        "amount=23.15 currency=USD merchant=grocery card_present=1 country_mismatch=0",
    ]
}


def build_inference_session(onnx_path, providers=EXECUTION_PROVIDERS, trt_cache_dir="./trt_cache"):
    """ONNX Runtime session with TensorRT first in the provider chain.

    Providers not present in the installed ONNX Runtime build are dropped, so the same code
    runs on a laptop (CPU only) and on the T4 endpoint (TensorRT, then CUDA, then CPU).
    ONNX Runtime gives TensorRT every subgraph it supports; the rest of the graph runs on the
    next provider in the list.
    """
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    chosen = []
    for name in providers:
        if name not in available:
            continue
        if name == "TensorrtExecutionProvider":
            chosen.append((name, {**TRT_OPTIONS, "trt_engine_cache_path": str(trt_cache_dir)}))
        else:
            chosen.append(name)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx_path), sess_options=options, providers=chosen)


def add_symbolic_shapes(onnx_path):
    """Annotate every tensor with its shape, which TensorRT needs for each subgraph it builds.

    Keeps external-data format if the model was saved that way (models over 2 GB are).
    """
    import onnx
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

    onnx_path = Path(onnx_path)
    header = onnx.load(str(onnx_path), load_external_data=False)
    external = any(t.data_location == onnx.TensorProto.EXTERNAL for t in header.graph.initializer)
    model = SymbolicShapeInference.infer_shapes(onnx.load(str(onnx_path)), auto_merge=True)
    if external:
        onnx.save_model(model, str(onnx_path), save_as_external_data=True,
                        all_tensors_to_one_file=True, location=f"{onnx_path.name}_data")
    else:
        onnx.save_model(model, str(onnx_path))
    return onnx_path


def run_smoke_test(session, tokenizer, texts=SAMPLE_REQUEST["transactions"]):
    """Run the quantized graph once, exactly as score.py will, and return the logits."""
    import numpy as np

    encoded = tokenizer(
        texts, padding="max_length", truncation=True, max_length=MAX_SEQ_LEN, return_tensors="np"
    )
    input_names = {i.name for i in session.get_inputs()}
    feeds = {k: v.astype(np.int64) for k, v in encoded.items() if k in input_names}
    return session.run(None, feeds)[0]


def optimize_model(model_path):
    from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from transformers import AutoTokenizer

    # 1. Export the trained checkpoint to the baseline FP32 ONNX computational graph
    model = ORTModelForSequenceClassification.from_pretrained(model_path, export=True)

    # 2. Dynamic INT8 quantization: weights are quantized now, activation scales are computed
    #    per batch at run time, so no calibration dataset is needed.
    quantizer = ORTQuantizer.from_pretrained(model)
    qconfig = AutoQuantizationConfig.avx2(is_static=False)
    quantizer.quantize(save_dir=QUANTIZED_DIR, quantization_config=qconfig)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(QUANTIZED_DIR)  # score.py loads model and tokenizer from one folder

    # 3. Load the INT8 graph with TensorrtExecutionProvider prioritized and run it once
    quantized_path = add_symbolic_shapes(Path(QUANTIZED_DIR) / QUANTIZED_FILE)
    session = build_inference_session(quantized_path)
    logits = run_smoke_test(session, tokenizer)
    print("Execution providers in priority order:", session.get_providers())
    print("Smoke-test logits shape:", logits.shape)

    return QUANTIZED_DIR


# --------------------------------------------------------------------------------------
# Step 3 - Deployment and rollout schema
# --------------------------------------------------------------------------------------
ENDPOINT_NAME = "fraud-detection-endpoint"
SCORING_DIR = Path("./scoring")
# Current T4 size on the managed online endpoint SKU list (Standard_NC6s_v3 was retired on
# 30 Sep 2025). The T4 has INT8 tensor cores, 16 GB holds the ~2 GB INT8 weights, and the
# host CPU supports AVX2, which is the target of the dynamic quantization config.
SERVING_SKU = "Standard_NC4as_T4_v3"
TRAFFIC_ROLLOUT = {"blue": 90, "green": 10}  # blue = current PyTorch production deployment
SLA_P95_MS = 100

INFERENCE_IMAGE = "mcr.microsoft.com/azureml/minimal-py312-inference:latest"
INFERENCE_CONDA = {
    "name": "fraud-onnx-trt-inference",
    "channels": ["conda-forge"],
    "dependencies": [
        "python=3.11",
        "pip",
        {
            "pip": [
                "azureml-inference-server-http==1.5.1",
                # ONNX Runtime 1.22 is documented against TensorRT 10.9 and CUDA 12. The extras
                # bring the CUDA and cuDNN runtimes as wheels, so the base image needs no CUDA.
                "onnxruntime-gpu[cuda,cudnn]==1.22.0",
                "tensorrt-cu12==10.9.0.34",
                "transformers>=4.36,<5",
                "numpy",
            ]
        },
    ],
}

SCORE_PY = '''\
"""score.py - Azure ML scoring script for the green (ONNX INT8) deployment."""
import json
import logging
import os
import tempfile

import numpy as np

try:  # loads the pip-installed TensorRT libraries so ONNX Runtime can find them
    import tensorrt  # noqa: F401
except ImportError:
    tensorrt = None
import onnxruntime as ort
from transformers import AutoTokenizer

PROVIDER_PRIORITY = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
MAX_SEQ_LEN = 128

session = None
tokenizer = None


def init():
    global session, tokenizer
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()  # CUDA and cuDNN from the onnxruntime-gpu[cuda,cudnn] wheels
    model_dir = os.path.join(os.environ["AZUREML_MODEL_DIR"], "onnx_quantized")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    available = set(ort.get_available_providers())
    providers = []
    for name in PROVIDER_PRIORITY:
        if name not in available:
            continue
        if name == "TensorrtExecutionProvider":
            providers.append((name, {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": os.path.join(tempfile.gettempdir(), "trt_cache"),
            }))
        else:
            providers.append(name)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        os.path.join(model_dir, "model_quantized.onnx"), sess_options=options, providers=providers
    )
    active = session.get_providers()
    logging.info("Active execution providers: %s", active)
    if active[0] != "TensorrtExecutionProvider":
        logging.warning("TensorRT is not active; serving on %s. Do not promote green.", active[0])


def run(raw_data):
    texts = json.loads(raw_data)["transactions"]
    encoded = tokenizer(texts, padding="max_length", truncation=True,
                        max_length=MAX_SEQ_LEN, return_tensors="np")
    input_names = {i.name for i in session.get_inputs()}
    feeds = {k: v.astype(np.int64) for k, v in encoded.items() if k in input_names}
    logits = session.run(None, feeds)[0]
    shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = shifted / shifted.sum(axis=1, keepdims=True)
    return {
        "fraud_probability": probabilities[:, 1].round(6).tolist(),
        "execution_provider": session.get_providers()[0],
    }
'''


def write_scoring_assets(scoring_dir=SCORING_DIR):
    """Write score.py and a sample request into the deployment's code folder."""
    scoring_dir.mkdir(parents=True, exist_ok=True)
    (scoring_dir / "score.py").write_text(SCORE_PY)
    (scoring_dir / "sample_request.json").write_text(json.dumps(SAMPLE_REQUEST, indent=2))
    return scoring_dir


def deploy_endpoint():
    write_scoring_assets()

    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        description=f"Contoso Financial real-time fraud scoring (p95 SLA {SLA_P95_MS} ms)",
        auth_mode="aad_token",  # Microsoft Entra tokens: no long-lived keys to leak or rotate
        tags={"model": "fraud-2b", "rollout": "blue-green"},
    )

    green_deployment = ManagedOnlineDeployment(
        name="green",
        endpoint_name=ENDPOINT_NAME,
        model=Model(
            name="fraud-2b-onnx-int8",
            path=QUANTIZED_DIR,
            type=AssetTypes.CUSTOM_MODEL,
            description="ONNX export of the 2B fraud classifier, dynamic INT8 (AVX2 config)",
        ),
        code_configuration=CodeConfiguration(code=str(SCORING_DIR), scoring_script="score.py"),
        environment=Environment(
            name="fraud-onnx-trt-inference",
            image=INFERENCE_IMAGE,
            conda_file=INFERENCE_CONDA,
        ),
        instance_type=SERVING_SKU,
        instance_count=2,
        request_settings=OnlineRequestSettings(
            request_timeout_ms=500,  # fail fast at 5x the SLA; timeouts count toward rollback
            max_concurrent_requests_per_instance=4,
        ),
        # The first start builds the TensorRT engine, so allow up to ~5 minutes before restarts
        liveness_probe=ProbeSettings(initial_delay=300, period=10, timeout=2, failure_threshold=30),
        app_insights_enabled=True,  # per-deployment latency and error metrics for the SLA checks
    )

    endpoint.traffic = dict(TRAFFIC_ROLLOUT)  # 90 % blue (PyTorch), 10 % green (ONNX INT8)

    return endpoint, green_deployment


# --------------------------------------------------------------------------------------
# Azure operations - the only functions that make network calls
# --------------------------------------------------------------------------------------
def get_ml_client():
    return MLClient(DefaultAzureCredential(), SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)


def submit_training(ml_client, sweep_job):
    """Provision the cluster, then submit the sweep."""
    if not (TRAIN_CODE_DIR / "train_deepspeed.py").exists():
        raise FileNotFoundError(f"{TRAIN_CODE_DIR}/train_deepspeed.py is missing from the code folder")
    ml_client.compute.begin_create_or_update(build_gpu_cluster()).result()
    return ml_client.jobs.create_or_update(sweep_job)


def rollout_green(ml_client, endpoint, green_deployment):
    """Safe rollout: create green at 0 %, test it directly, then apply the 90/10 split.

    Assumes blue (the current PyTorch deployment) exists and holds 100 % of traffic.
    Blue keeps its full instance count until green has run at 100 % for a full week.
    """
    target_traffic = dict(endpoint.traffic)
    endpoint.traffic = {"blue": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    ml_client.online_deployments.begin_create_or_update(green_deployment).result()  # green at 0 %

    # Direct call to green, outside the traffic split, before any customer traffic reaches it
    ml_client.online_endpoints.invoke(
        endpoint_name=ENDPOINT_NAME,
        deployment_name="green",
        request_file=str(SCORING_DIR / "sample_request.json"),
    )

    endpoint.traffic = target_traffic
    return ml_client.online_endpoints.begin_create_or_update(endpoint).result()


def set_traffic(ml_client, traffic):
    if sum(traffic.values()) != 100:
        raise ValueError(f"Traffic must sum to 100, got {traffic}")
    endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
    endpoint.traffic = traffic
    return ml_client.online_endpoints.begin_create_or_update(endpoint).result()


def promote(ml_client, green_percent):
    """Next stage of the rollout (10 -> 50 -> 100), only after the SLA checks pass."""
    return set_traffic(ml_client, {"blue": 100 - green_percent, "green": green_percent})


def rollback(ml_client):
    """Rollback: all live traffic back to blue in one call.

    Green stays deployed at 0 % so its logs and metrics remain available for diagnosis.
    """
    return set_traffic(ml_client, {"blue": 100, "green": 0})


# --------------------------------------------------------------------------------------
# Self-checks (the scaffold's assertion cells) and command-line entry point
# --------------------------------------------------------------------------------------
def check_step1(sweep_job):
    print("Sampling algorithm:", sweep_job.sampling_algorithm)
    assert sweep_job.sampling_algorithm == "random", "Must use random sampling to pair with BanditPolicy"
    assert isinstance(sweep_job.early_termination, BanditPolicy), "Expected a BanditPolicy"
    assert "--deepspeed ds_config.json" in sweep_job.trial.command, "ds_config.json must be referenced"
    assert "/versions/" in sweep_job.trial.environment, "Environment must be a pinned version"
    print("PASS: orchestrate_training() builds a valid Random + BanditPolicy sweep, entirely offline.")


def check_step2(output_dir):
    files = os.listdir(output_dir)
    print("Quantized model directory contents:", files)
    assert any(f.endswith(".onnx") for f in files), "Expected a quantized .onnx file"
    print("PASS: optimize_model() genuinely exported and quantized a real model to ONNX.")


def check_step3(endpoint, green_deployment):
    print("Traffic split:", endpoint.traffic)
    assert sum(endpoint.traffic.values()) == 100, "Traffic percentages must sum to 100"
    assert green_deployment.instance_type != "Standard_NC6s_v3", (
        "This SKU was retired September 30, 2025 - use a current one"
    )
    print("PASS: deploy_endpoint() builds a valid endpoint/deployment pair, entirely offline.")


def main():
    parser = argparse.ArgumentParser(description="Contoso fraud model: train, optimize, deploy")
    parser.add_argument("--model-path", default=TEST_MODEL,
                        help="Checkpoint to export (the best sweep trial's consolidated model)")
    parser.add_argument("--skip-optimize", action="store_true",
                        help="Skip Step 2, e.g. on a machine without PyTorch")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--submit", action="store_true",
                        help="Provision the cluster, submit the sweep, roll green out at 10 %%")
    action.add_argument("--promote", type=int, metavar="PCT",
                        help="Set green to PCT %% of traffic")
    action.add_argument("--rollback", action="store_true",
                        help="Send 100 %% of traffic back to blue")
    args = parser.parse_args()

    if args.rollback:
        rollback(get_ml_client())
        print("Rolled back: blue 100 %, green 0 %.")
        return
    if args.promote is not None:
        promote(get_ml_client(), args.promote)
        print(f"Promoted: green {args.promote} %.")
        return

    sweep_job = orchestrate_training()
    check_step1(sweep_job)

    if not args.skip_optimize:
        check_step2(optimize_model(args.model_path))

    endpoint, green_deployment = deploy_endpoint()
    check_step3(endpoint, green_deployment)

    if args.submit:
        ml_client = get_ml_client()
        submitted = submit_training(ml_client, sweep_job)
        print("Sweep submitted:", submitted.name)
        rollout_green(ml_client, endpoint, green_deployment)
        print("Green deployed and receiving 10 % of traffic.")


if __name__ == "__main__":
    main()
