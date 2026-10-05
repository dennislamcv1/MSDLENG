#!/usr/bin/env python
# coding: utf-8

# # Deep Learning Environment & Baseline Model — Starter Notebook
# 
# Complete the TODOs in each section below, following the assignment's
# Step 1 through Step 4. Step 5 (your written justification) goes in the
# final Markdown cell at the bottom of this notebook.
# 
# Run cells in order as you complete them.

# In[ ]:


# Imports for PyTorch and Azure ML SDK v2
import os
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow

from azure.core.exceptions import ClientAuthenticationError
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import AmlCompute, Input, PyTorchDistribution
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.identity import DefaultAzureCredential, DeviceCodeCredential


# ## PART 1: PyTorch Engineering
# 
# **Step 1 — Architect the Model** and **Step 2 — Construct the Training
# Loop with Telemetry** both go in this section.
# 
# *In a real project, this would live in a separate `train.py` file.*

# In[ ]:


def train_model():
    """
    This function defines and trains the PyTorch model.
    It is called by the Azure ML command job.
    """

    class BaselineModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Step 1: two fully connected layers (10 features -> 50 hidden -> 2 outputs)
            self.fc1 = nn.Linear(10, 50)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(50, 2)

        def forward(self, x):
            # Step 1: Linear -> ReLU -> Linear
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # Step 2: initialize the model, loss function, and optimizer
    model = BaselineModel()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # Dummy data for blueprint validation
    inputs = torch.randn(16, 10)
    targets = torch.randn(16, 2)

    for epoch in range(5):
        # Step 2: the four core training steps
        optimizer.zero_grad()                 # 1. clear old gradients
        outputs = model(inputs)               # 2. forward pass
        loss = criterion(outputs, targets)    # 3. calculate loss
        loss.backward()                       # 4. backpropagate
        optimizer.step()                      # 5. update weights

        # Step 2: telemetry - only the primary process (rank 0) writes logs
        if os.environ.get("RANK", "0") == "0":
            mlflow.log_metric("train_loss", loss.item(), step=epoch)

    return model


if __name__ == "__main__":
    # Optional local smoke test (runs without Azure; metrics go to a local mlruns/ folder)
    train_model()


# ## PART 2: Azure ML Infrastructure Blueprint
# 
# **Step 3 — Define the Cloud Infrastructure Blueprint** and **Step 4 —
# Package the Command Job** both go in this section.
# 
# *In a real project, this would live in a separate `provision.py` file
# or a dedicated infrastructure notebook.*

# In[ ]:


def provision_infrastructure():
    """
    This function defines and configures the Azure ML infrastructure.
    """

    # Step 3: workspace identifiers (replace with your values or set as env vars)
    subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "<SUBSCRIPTION_ID>")
    resource_group = os.environ.get("AZURE_RESOURCE_GROUP", "<RESOURCE_GROUP>")
    workspace_name = os.environ.get("AZUREML_WORKSPACE_NAME", "<WORKSPACE_NAME>")

    # Step 3: DefaultAzureCredential first; DeviceCodeCredential as a fallback
    # for local or headless environments where the primary chain cannot resolve.
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except ClientAuthenticationError:
        credential = DeviceCodeCredential()

    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

    # Step 3: GPU cluster that scales to zero when idle
    gpu_cluster = AmlCompute(
        name="gpu-cluster",
        size="Standard_NCasT4_v3",
        min_instances=0,                  # no idle GPU cost
        max_instances=4,                  # upper bound on burst spend
        idle_time_before_scale_down=120,  # seconds idle before nodes are released
    )
    # In a live environment: ml_client.compute.begin_create_or_update(gpu_cluster).result()

    # Step 4: mock data input, mounted (streamed) rather than downloaded
    my_data_input = Input(
        type=AssetTypes.URI_FOLDER,
        path="azureml://datastores/mystore/paths/train_data/",
        mode=InputOutputModes.MOUNT,
    )

    # Step 4: command job payload
    training_job = command(
        code="./src",                                   # folder containing train.py
        command="python train.py --data ${{inputs.training_data}}",
        inputs={"training_data": my_data_input},
        environment="azureml:AzureML-ACPT-pytorch-2.2-cuda12.1@latest",
        compute="gpu-cluster",
        instance_count=1,
        distribution=PyTorchDistribution(process_count_per_instance=1),
        display_name="baseline-model-training",
        experiment_name="baseline-model",
    )

    # In a live environment, you would uncomment the following line to submit the job:
    # submitted_job = ml_client.jobs.create_or_update(training_job)
    return ml_client, gpu_cluster, training_job


# ## Step 5 — Architectural Justification
# 
# **Optimizer choice: AdamW.** I chose AdamW because it fixes a well-known flaw in standard Adam: in Adam, L2 regularization is folded into the gradient and then rescaled by the adaptive per-parameter learning rates, so the effective regularization strength becomes inconsistent across parameters. AdamW *decouples* weight decay from the gradient update and applies it directly to the weights. This makes `weight_decay` a clean, independently tunable hyperparameter and generally yields better generalization, which matters for a model of customer purchase behavior that must perform on unseen customers. AdamW also keeps the fast, stable convergence of adaptive methods with little tuning (a learning rate of 1e-3 and weight decay of 0.01 are sensible defaults), and it is the de facto standard in modern deep learning, so the whole AI division can adopt this template with well-understood behavior. Lion is more memory-efficient because it stores only one momentum term, but it is more sensitive to learning-rate and weight-decay tuning. For a baseline template meant to be reused broadly, AdamW's reliability outweighs Lion's memory savings.
# 
# **Why `min_instances=0` is non-negotiable.** GPU nodes such as Standard_NCasT4_v3 are billed for every second they are allocated, whether or not they are doing useful work. With `min_instances=0`, the cluster holds no nodes while idle, so cost is incurred only while jobs run. Combined with `idle_time_before_scale_down`, nodes are released shortly after a job finishes. A cluster with `min_instances` above zero would burn budget around the clock (nights, weekends, holidays), and across a division running many experiments this waste compounds quickly into unplanned spend. Scale-to-zero also makes costs proportional to actual usage, which is easier to forecast, budget, and charge back to teams. Paired with `max_instances=4`, which caps burst spend, it gives the enterprise elasticity without runaway cost. The only trade-off is a few minutes of node start-up latency on the first job, which is a small price for training workloads that are not latency-sensitive.
