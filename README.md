# Carl
CUDA Load Tester for Kubernetes

Carl is designed to provide flexible gpu load testing for your kubernetes cluster. It accomplishes this by generating a deployment of pods that each attempt to use a monte-carlo approximation of Pi based on the number of input blocks fed to it. More blocks == better approximation and more GPU utilization. 

## Requirements

- At least 1 Nvidia GPU enabled worker node, Maxwell generation (compute capability 5.0) or later.
- An Nvidia driver that supports CUDA 12.9 or newer (driver 575+; older 525+ drivers work via CUDA minor version compatibility).
- The Nvidia device plugin (or GPU Operator) deployed to your cluster so that `nvidia.com/gpu` is schedulable: https://github.com/NVIDIA/k8s-device-plugin

The container image is built from [`source/`](source/) on top of `nvidia/cuda:12.9.1-base-ubuntu24.04`
and is published to `ghcr.io/chandan-m-n/carl` by the [build workflow](.github/workflows/build.yaml)
on every push to `main` (`latest`, `sha-<commit>`) and on `v*` tags.

## Usage

### Continuous load (Deployment)

```
kubectl apply -f https://raw.githubusercontent.com/Chandan-M-N/carl/main/carl.yaml
```

Or download and modify the deployment yaml before applying:

```
curl -o carl.yaml https://raw.githubusercontent.com/Chandan-M-N/carl/main/carl.yaml
```

By default Carl only deploys a single pod since you need at least one GPU enabled worker to utilize this workload. After deployment you can scale up the number of replicas to ensure all nodes can accept your gpu workload:

```
kubectl scale deployment carl --replicas=<number of GPUs>
kubectl logs -l name=carl-gpu-deploy -f
```

### One-shot test (Job)

[`carl-job.yaml`](carl-job.yaml) runs the test a fixed number of times and then completes, which is handy for
validating a cluster:

```
kubectl apply -f https://raw.githubusercontent.com/Chandan-M-N/carl/main/carl-job.yaml
kubectl wait --for=condition=complete job/carl-test --timeout=10m
kubectl logs -l job-name=carl-test
```

Raise `completions`/`parallelism` in the Job to test several GPUs at once.

### Configuration

| Setting | Where | Default | Description |
|---|---|---|---|
| blocks | container `args` | `512` (`16384` in the manifests) | CUDA blocks of 512 threads each; more blocks == more GPU work and memory |
| `ITERATIONS` | env | `0` | Number of runs before exiting, `0` runs forever |
| `SLEEP_SECONDS` | env | `60` | Pause between runs |

The script exits non-zero if a CUDA error occurs, so failures surface as pod restarts or failed Jobs.

### Tolerations

The default taints Carl can tolerate are:
```
      tolerations:
      - key: "dedicated"
        operator: "Equal"
        value: "gpu-worker"
        effect: "NoExecute"
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
```
        
Carl requests only a single gpu for each pod, multi-gpu per pod will be tested later. 

## Building the image yourself

```
docker build -t <registry>/carl:<tag> source
```

Override `--build-arg CUDA_VERSION=...` or `--build-arg CUDA_ARCHS="80 86 90"` to change the CUDA toolkit
or the GPU architectures the binary is compiled for.
