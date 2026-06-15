# Kubernetes Node Certificate Expiration on AWS EKS

## Purpose

This guide explains Kubernetes node certificate expiration, how it applies to AWS EKS, how to check certificate health, and the practical approaches for managed node groups, self-managed nodes, Bottlerocket, Fargate, and emergency recovery.

## Quick Summary

In AWS EKS, the Kubernetes control plane certificates are managed by AWS. For worker nodes, kubelet client certificate rotation is normally handled automatically when nodes are bootstrapped correctly. The main operational risk is with self-managed nodes, custom AMIs, disabled kubelet rotation, broken bootstrap configuration, or nodes that have been running for a long time without refresh.

Recommended approach for most teams:

1. Use EKS managed node groups where possible.
2. Keep kubelet client certificate rotation enabled.
3. Rotate or refresh nodes regularly.
4. Monitor kubelet certificate expiration metrics.
5. Alert before certificates are close to expiry.
6. Avoid relying only on manual CSR approval as a long-term strategy.

## What Certificates Exist in EKS?

### EKS Control Plane Certificates

AWS manages EKS control plane certificates. These include the API server and internal control plane certificates.

You do not SSH into or rotate EKS control plane certificates yourself.

Operational responsibility:

- AWS manages control plane certificate lifecycle.
- You manage cluster version upgrades and worker node lifecycle.
- You monitor client access and node connectivity.

### Worker Node Certificates

Each Kubernetes node runs `kubelet`. The kubelet needs certificates to authenticate with the Kubernetes API server.

Important kubelet certificate types:

| Certificate | Purpose | Typical Location |
|---|---|---|
| Kubelet client certificate | Kubelet authenticates to API server | `/var/lib/kubelet/pki/kubelet-client-current.pem` |
| Kubelet serving certificate | API server or clients connect to kubelet HTTPS endpoint | `/var/lib/kubelet/pki/kubelet-server-current.pem` or generated dynamically |
| CA bundle | Trusts cluster CA | `/etc/kubernetes/pki/ca.crt` or bootstrap-provided location |

The most important one for node readiness is usually the kubelet client certificate.

## What Happens When Kubelet Certificates Expire?

If a kubelet client certificate expires and cannot rotate, the node may fail to authenticate to the API server.

Common symptoms:

- Node becomes `NotReady`.
- `kubectl logs`, `kubectl exec`, or metrics collection may fail.
- New pods may not schedule on affected nodes.
- Kubelet logs show `x509: certificate has expired` or authentication failures.
- CertificateSigningRequests may remain `Pending`.
- Node heartbeats stop updating.

Example errors:

```text
x509: certificate has expired or is not yet valid
Unable to authenticate the request due to an error: x509: certificate has expired
certificate rotation is enabled but no approved certificate is available
```

## EKS Approaches by Node Type

## 1. EKS Managed Node Groups

Managed node groups are the preferred approach for most production EKS clusters.

AWS manages the node group lifecycle integration, and the standard EKS bootstrap process configures kubelet correctly.

Recommended actions:

- Use managed node groups for normal Linux worker nodes.
- Keep node AMIs current.
- Regularly update or replace node groups.
- Monitor certificate expiration metrics.
- Avoid manually changing kubelet bootstrap files unless required.

Typical certificate rotation expectation:

- Kubelet client certificate rotation should be enabled through standard EKS bootstrap.
- Kubelet requests a new certificate before expiry.
- Kubernetes certificate controller approves valid kubelet client certificate renewal requests.

Best operational pattern:

```text
Managed node group + regular node refresh + monitoring = lowest operational risk
```

## 2. Self-Managed Worker Nodes

Self-managed nodes need more care because you own more of the bootstrap and lifecycle.

Use self-managed nodes only when you need custom behavior that managed node groups cannot provide.

Key requirements:

- Use the official EKS optimized AMI where possible.
- Use `/etc/eks/bootstrap.sh` correctly.
- Keep `rotateCertificates: true` enabled.
- Ensure kubelet has access to bootstrap credentials during first join.
- Ensure CSR approval is working.
- Replace old nodes before certificate or OS lifecycle risk becomes high.

Kubelet config should include:

```yaml
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
rotateCertificates: true
serverTLSBootstrap: true
```

Depending on EKS version and AMI, the effective kubelet config may live under:

```text
/etc/kubernetes/kubelet/kubelet-config.json
/var/lib/kubelet/config.yaml
```

Check on the node:

```bash
sudo grep -R "rotateCertificates" /etc/kubernetes /var/lib/kubelet 2>/dev/null
sudo systemctl cat kubelet
sudo journalctl -u kubelet --no-pager | grep -i certificate
```

## 3. Bottlerocket Nodes

Bottlerocket is commonly used with EKS managed node groups and handles node configuration differently from Amazon Linux.

Recommended actions:

- Prefer Bottlerocket with managed node groups.
- Use EKS-supported Bottlerocket variants.
- Update Bottlerocket regularly.
- Monitor node readiness and kubelet certificate metrics.
- Replace nodes through managed node group update workflows.

For Bottlerocket, avoid assuming Amazon Linux file paths always exist. Use Bottlerocket-supported admin/container access and EKS node group updates for lifecycle operations.

## 4. EKS Fargate

EKS Fargate does not expose worker nodes for you to manage.

Operational responsibility:

- AWS manages the underlying compute infrastructure.
- You do not rotate node kubelet certificates.
- You monitor pod health, scheduling, and application behavior.

For Fargate workloads, certificate expiration concern is mostly about application certificates, ingress certificates, service mesh certificates, or external TLS, not kubelet node certificates.

## How to Check Node Certificate Expiration

## Check from a Worker Node

SSH or SSM into a node and run:

```bash
sudo openssl x509 \
  -in /var/lib/kubelet/pki/kubelet-client-current.pem \
  -noout \
  -subject \
  -issuer \
  -dates
```

Example output:

```text
subject=O = system:nodes, CN = system:node:ip-10-0-1-20.ec2.internal
issuer=CN = kubernetes
notBefore=Jun 10 10:00:00 2026 GMT
notAfter=Jun 10 10:00:00 2027 GMT
```

Check only the expiry date:

```bash
sudo openssl x509 \
  -in /var/lib/kubelet/pki/kubelet-client-current.pem \
  -noout \
  -enddate
```

## Check Multiple Nodes with SSM

If your nodes use AWS Systems Manager, use SSM Run Command to run the OpenSSL check across nodes.

Example command to run on each node:

```bash
sudo openssl x509 -in /var/lib/kubelet/pki/kubelet-client-current.pem -noout -subject -enddate
```

This is useful for audit checks across large node fleets.

## Check CertificateSigningRequests

Run:

```bash
kubectl get csr
```

Look for pending or failed requests:

```bash
kubectl get csr | grep -E "Pending|Denied|Failed"
```

Describe a CSR:

```bash
kubectl describe csr <csr-name>
```

Approve a valid pending CSR manually:

```bash
kubectl certificate approve <csr-name>
```

Manual approval should be used carefully. Do not approve unknown CSRs blindly.

## Check Node Readiness and Heartbeats

```bash
kubectl get nodes -o wide
```

Check node conditions:

```bash
kubectl describe node <node-name>
```

Check recent Ready condition timestamps:

```bash
kubectl get nodes -o custom-columns=NAME:.metadata.name,READY:.status.conditions[-1].type,STATUS:.status.conditions[-1].status,LAST_HEARTBEAT:.status.conditions[-1].lastHeartbeatTime
```

Note: JSONPath and custom-columns using condition indexes can be imperfect because condition ordering is not always something you should build automation around. For automation, parse `.status.conditions[]` and select `type == Ready`.

## Monitoring with Prometheus

Kubelet exposes certificate manager metrics.

Useful metrics:

```text
kubelet_certificate_manager_client_expiration_seconds
kubelet_certificate_manager_server_expiration_seconds
```

These values are Unix timestamps for certificate expiration.

Prometheus alert example:

```yaml
groups:
  - name: kubelet-certificates
    rules:
      - alert: KubeletClientCertificateExpiresSoon
        expr: (kubelet_certificate_manager_client_expiration_seconds - time()) < 2592000
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Kubelet client certificate expires within 30 days"
          description: "Node {{ $labels.instance }} has a kubelet client certificate expiring within 30 days."

      - alert: KubeletClientCertificateExpiresCritical
        expr: (kubelet_certificate_manager_client_expiration_seconds - time()) < 604800
        for: 15m
        labels:
          severity: critical
        annotations:
          summary: "Kubelet client certificate expires within 7 days"
          description: "Node {{ $labels.instance }} has a kubelet client certificate expiring within 7 days. Replace or repair the node immediately."
```

Recommended alert thresholds:

| Severity | Threshold | Action |
|---|---:|---|
| Info | 60 days | Review node age and rotation status |
| Warning | 30 days | Plan node refresh or repair rotation |
| Critical | 7 days | Replace node or fix CSR rotation urgently |
| Emergency | Expired | Drain if possible, replace, or re-bootstrap |

## Node Rotation Strategy

Even if certificate rotation works, regular node replacement is still a good practice.

Benefits:

- Reduces certificate aging risk.
- Applies OS and kernel patches.
- Picks up new EKS optimized AMIs.
- Reduces drift from manual changes.
- Validates autoscaling and workload disruption settings.

Recommended cadence:

| Environment | Suggested Node Refresh Cadence |
|---|---:|
| Development | Every 1 to 3 months |
| Staging | Every 1 to 2 months |
| Production | Every 1 to 3 months, depending on risk and patch policy |

For managed node groups:

```bash
aws eks update-nodegroup-version \
  --cluster-name <cluster-name> \
  --nodegroup-name <nodegroup-name>
```

For blue/green node group replacement:

1. Create a new node group with the desired AMI or launch template.
2. Confirm new nodes join and become Ready.
3. Cordon old nodes.
4. Drain old nodes safely.
5. Delete the old node group after workload validation.

Example drain:

```bash
kubectl cordon <old-node-name>
kubectl drain <old-node-name> \
  --ignore-daemonsets \
  --delete-emptydir-data \
  --grace-period=60
```

## Emergency Recovery

Use this section when nodes are already `NotReady` because kubelet certificates expired or rotation broke.

## Option A: Replace the Node

This is usually the cleanest approach in EKS.

For managed node groups:

1. Increase desired capacity or ensure autoscaler can add replacement nodes.
2. Terminate the unhealthy EC2 instance.
3. Let the node group replace it.
4. Confirm replacement node joins the cluster.
5. Validate workloads reschedule.

Commands:

```bash
kubectl get nodes -o wide
aws ec2 terminate-instances --instance-ids <instance-id>
```

This is preferred when workloads are replicated and disruption budgets are healthy.

## Option B: Manually Approve Pending Kubelet CSRs

Use only if you have verified the CSR is legitimate.

Check pending CSRs:

```bash
kubectl get csr
```

Describe the CSR:

```bash
kubectl describe csr <csr-name>
```

Verify:

- Requester is expected.
- Node name matches an expected node.
- Groups include expected kubelet bootstrap or node groups.
- Usages are appropriate for kubelet client auth.

Approve:

```bash
kubectl certificate approve <csr-name>
```

Then check node status:

```bash
kubectl get nodes
```

## Option C: Re-bootstrap the Node

For self-managed nodes, re-bootstrap may be needed if certificate state is broken.

High-level steps:

1. Cordon and drain the node if possible.
2. Stop kubelet.
3. Back up kubelet PKI files.
4. Remove expired kubelet client certificate files.
5. Re-run EKS bootstrap with correct cluster parameters.
6. Start kubelet.
7. Verify CSR approval and node readiness.

Example investigation commands:

```bash
sudo systemctl stop kubelet
sudo ls -l /var/lib/kubelet/pki/
sudo openssl x509 -in /var/lib/kubelet/pki/kubelet-client-current.pem -noout -dates
```

Do not delete kubelet PKI files blindly on production nodes. Prefer replacement when possible.

## Common Root Causes

| Root Cause | Description | Fix |
|---|---|---|
| Kubelet rotation disabled | `rotateCertificates` is false or missing | Enable rotation and restart/recreate nodes |
| Broken bootstrap | Node joined with nonstandard config | Re-bootstrap or replace node |
| Pending CSRs not approved | Certificate renewal request exists but is not approved | Fix CSR approval or approve valid CSRs |
| Very old nodes | Nodes have not been refreshed for a long time | Rotate node group |
| Custom AMI drift | Kubelet config differs from EKS expectations | Rebuild AMI from EKS optimized baseline |
| Time sync issue | Node clock is wrong | Fix chrony/NTP and verify time |
| IAM/auth issue | Node cannot authenticate during bootstrap | Fix node IAM role and `aws-auth` or access entries |

## EKS Authentication Note

Modern EKS clusters may use EKS access entries, while older clusters commonly use the `aws-auth` ConfigMap.

Worker nodes must be authorized to join the cluster.

Check `aws-auth` if your cluster uses it:

```bash
kubectl -n kube-system get configmap aws-auth -o yaml
```

For access entries, check with AWS CLI:

```bash
aws eks list-access-entries --cluster-name <cluster-name>
```

If node IAM role authorization is broken, nodes may fail to join or rejoin, which can look similar to certificate problems during recovery.

## Security Guidance for CSR Approval

Do not automatically approve every pending CSR.

Before approval, verify:

- The CSR is for a real node you own.
- The common name follows `system:node:<node-name>`.
- The organization is `system:nodes`.
- The requested usages are expected.
- The node identity and instance are legitimate.

A suspicious CSR can allow unauthorized node identity if approved incorrectly.

## Recommended Production Runbook

## Daily or Weekly Checks

```bash
kubectl get nodes
kubectl get csr
```

Review:

- Nodes not Ready.
- Pending CSRs.
- Nodes older than your refresh policy.
- Certificate expiration alerts.

## Monthly Checks

1. Review node group AMI versions.
2. Review EKS cluster version support window.
3. Confirm managed node group update strategy.
4. Check Prometheus certificate alerts.
5. Test one controlled node replacement in non-production.

## Before Expiry Alert Becomes Critical

1. Identify affected node.
2. Check whether kubelet has requested a new certificate.
3. Check kubelet logs for rotation errors.
4. Confirm node IAM role and cluster authorization.
5. Prefer replacing the node if it is old or misconfigured.
6. Manually approve only verified CSRs.

## Practical Decision Matrix

| Situation | Best Approach |
|---|---|
| Managed node group, one old node close to expiry | Replace or recycle the node |
| Managed node group, many nodes close to expiry | Perform managed node group update or blue/green replacement |
| Self-managed node with disabled rotation | Create fixed AMI/config and replace nodes |
| Pending valid kubelet renewal CSR | Approve CSR and investigate why auto-approval failed |
| Expired certificate and node is NotReady | Replace node if workload allows |
| Custom AMI with unknown kubelet config | Rebuild from EKS optimized AMI baseline |
| Fargate workload | No node certificate action required |

## Useful Commands

Set variables:

```bash
CLUSTER_NAME="<cluster-name>"
AWS_REGION="<region>"
NODEGROUP_NAME="<nodegroup-name>"
```

List nodes:

```bash
kubectl get nodes -o wide
```

List CSRs:

```bash
kubectl get csr
```

Check node details:

```bash
kubectl describe node <node-name>
```

Check kubelet logs on node:

```bash
sudo journalctl -u kubelet --since "2 hours ago" --no-pager
```

Check kubelet client certificate expiry:

```bash
sudo openssl x509 -in /var/lib/kubelet/pki/kubelet-client-current.pem -noout -subject -issuer -dates
```

List EKS node groups:

```bash
aws eks list-nodegroups \
  --cluster-name "$CLUSTER_NAME" \
  --region "$AWS_REGION"
```

Describe node group:

```bash
aws eks describe-nodegroup \
  --cluster-name "$CLUSTER_NAME" \
  --nodegroup-name "$NODEGROUP_NAME" \
  --region "$AWS_REGION"
```

Update managed node group:

```bash
aws eks update-nodegroup-version \
  --cluster-name "$CLUSTER_NAME" \
  --nodegroup-name "$NODEGROUP_NAME" \
  --region "$AWS_REGION"
```

## Final Recommendations

For AWS EKS, treat node certificate expiration as a node lifecycle and monitoring problem, not as a manual certificate maintenance task.

Best long-term setup:

1. Use managed node groups or Bottlerocket managed node groups.
2. Keep nodes replaceable and avoid long-lived pets.
3. Enable and verify kubelet certificate rotation for self-managed nodes.
4. Alert on kubelet certificate expiration using Prometheus.
5. Refresh nodes regularly.
6. Prefer node replacement over manual certificate surgery.
7. Keep EKS cluster versions, node AMIs, launch templates, and bootstrap configuration current.

If you follow those practices, kubelet certificate expiration should be rare, visible before it becomes urgent, and recoverable through normal node replacement workflows.
