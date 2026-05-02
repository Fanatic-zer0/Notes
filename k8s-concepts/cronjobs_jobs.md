# Jobs & CronJobs in Kubernetes

## Overview

| Resource | Purpose | Restarts |
|----------|---------|----------|
| **Job** | Run a task to completion | Until success or maxRetries |
| **CronJob** | Scheduled recurring tasks | Per schedule |

Jobs are designed for **batch workloads** - tasks that run, complete, and then stop.

## Job

### Basic Job

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration
spec:
  # Job behavior
  completions: 1           # How many pods must succeed
  parallelism: 1           # How many pods run simultaneously
  backoffLimit: 3          # Max retries before job fails
  activeDeadlineSeconds: 600  # Job timeout (10 min)
  ttlSecondsAfterFinished: 3600  # Auto-delete job after 1 hour
  
  template:
    spec:
      restartPolicy: Never  # OnFailure or Never (Always not allowed)
      
      containers:
      - name: migration
        image: myapp:latest
        command: ["./migrate", "--direction=up", "--all"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
```

### Job Execution Flow

```
kubectl apply -f db-migration-job.yaml
     │
     ├─→ 1. Job resource created in etcd
     │
     ├─→ 2. Job controller creates Pod
     │       db-migration-xxxxx
     │
     ├─→ 3. Scheduler places pod on a node
     │
     ├─→ 4. Pod runs migration
     │       ./migrate --direction=up --all
     │
     ├─→ 5a. Pod exits with code 0 (SUCCESS)
     │         Job status: Complete
     │         Pod remains (for log inspection)
     │
     └─→ 5b. Pod exits with non-zero (FAILURE)
              backoffLimit=3 → Retry up to 3 times
              Retry with exponential backoff: 10s, 20s, 40s
              If 3 failures → Job status: Failed
```

## Parallel Jobs

### Fixed Completions (Work Queue)

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: image-processing
spec:
  completions: 100    # Process 100 images total
  parallelism: 10     # 10 at a time
  
  template:
    spec:
      restartPolicy: OnFailure
      containers:
      - name: processor
        image: image-processor:latest
        env:
        - name: JOB_INDEX
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
```

**Progress:**
```
Start: 10 pods running (processing images 1-10)
       Pod 3 finishes → Pod 11 starts
       Pod 7 finishes → Pod 12 starts
       ...
End:   100/100 completions, all pods done
```

### Indexed Job (Each Pod Gets Unique Index)

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: data-sharding
spec:
  completions: 5
  parallelism: 5
  completionMode: Indexed  # Each pod gets unique index 0-4
  
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: worker
        image: data-processor:latest
        command:
        - sh
        - -c
        - |
          # JOB_COMPLETION_INDEX is auto-injected (0, 1, 2, 3, 4)
          echo "Processing shard $JOB_COMPLETION_INDEX of 5"
          ./process-shard.sh --shard=$JOB_COMPLETION_INDEX --total=5
```

### Work Queue Pattern

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: queue-consumer
spec:
  parallelism: 5    # Run 5 workers
  # No completions = run until all workers exit 0
  
  template:
    spec:
      restartPolicy: Never
      containers:
      - name: worker
        image: worker:latest
        command:
        - sh
        - -c
        - |
          # Worker processes messages until queue is empty
          while true; do
            msg=$(redis-cli -h redis BLPOP work-queue 10)
            if [ -z "$msg" ]; then
              echo "Queue empty, exiting"
              exit 0
            fi
            process_message "$msg"
          done
```

## CronJob

### Basic CronJob

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: db-backup
spec:
  # Cron expression: minute hour day-of-month month day-of-week
  schedule: "0 2 * * *"      # Every day at 2:00 AM
  timeZone: "America/New_York"  # Explicit timezone (K8s 1.27+)
  
  # CronJob-specific settings
  concurrencyPolicy: Forbid         # Allow, Forbid, or Replace
  failedJobsHistoryLimit: 3         # Keep last 3 failed job records
  successfulJobsHistoryLimit: 1     # Keep last 1 successful job record
  startingDeadlineSeconds: 300      # If 5 min late, skip this run
  suspend: false                    # Temporarily disable
  
  jobTemplate:
    spec:
      backoffLimit: 2
      activeDeadlineSeconds: 3600   # Job timeout: 1 hour
      ttlSecondsAfterFinished: 86400 # Auto-delete after 1 day
      
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: backup
            image: backup-tool:latest
            command:
            - sh
            - -c
            - |
              DATE=$(date +%Y%m%d-%H%M%S)
              pg_dump -h postgres -U $DB_USER $DB_NAME | \
                gzip | \
                aws s3 cp - s3://$S3_BUCKET/backups/db-$DATE.sql.gz
              echo "Backup completed: db-$DATE.sql.gz"
            env:
            - name: DB_USER
              valueFrom:
                secretKeyRef:
                  name: db-credentials
                  key: username
            - name: DB_NAME
              value: "production"
            - name: S3_BUCKET
              valueFrom:
                configMapKeyRef:
                  name: backup-config
                  key: s3-bucket
            resources:
              requests:
                cpu: 200m
                memory: 256Mi
```

## Cron Schedule Examples

```
┌─────────────── minute (0-59)
│ ┌───────────── hour (0-23)
│ │ ┌─────────── day of month (1-31)
│ │ │ ┌───────── month (1-12)
│ │ │ │ ┌─────── day of week (0-6, 0=Sunday)
│ │ │ │ │
* * * * *

Common patterns:
"* * * * *"          Every minute
"*/5 * * * *"        Every 5 minutes
"0 * * * *"          Every hour (at :00)
"0 9 * * 1-5"        9 AM, Mon-Fri (weekdays)
"0 0 * * 0"          Midnight Sunday
"0 2 * * *"          2 AM daily
"0 0 1 * *"          Midnight, 1st of each month
"0 0 1 1 *"          Midnight, Jan 1st (New Year)
"*/15 * * * *"       Every 15 minutes
"0 0,12 * * *"       Midnight and noon daily
"@hourly"            Shorthand for 0 * * * *
"@daily"             Shorthand for 0 0 * * *
"@weekly"            Shorthand for 0 0 * * 0
"@monthly"           Shorthand for 0 0 1 * *
```

## Concurrency Policy

```
CronJob schedule: "*/2 * * * *" (every 2 min)
Job takes 5 minutes to complete

Timeline:
  :00 → Job1 starts
  :02 → Job2 should start
  :04 → Job3 should start
  :05 → Job1 finishes

concurrencyPolicy: Allow (default)
  :00 Job1 starts
  :02 Job2 starts (Job1 still running)
  :04 Job3 starts (Job1, Job2 still running)
  Result: Multiple jobs running simultaneously

concurrencyPolicy: Forbid
  :00 Job1 starts
  :02 Skip (Job1 still running)
  :04 Skip (Job1 still running)
  :06 Job4 starts (Job1 finished at :05)
  Result: Prevents overlap

concurrencyPolicy: Replace
  :00 Job1 starts
  :02 Kill Job1, start Job2
  :04 Kill Job2, start Job3
  Result: Always runs fresh job, kills previous
```

## Advanced Job Patterns

### Pre/Post Hooks

```yaml
# Use init containers for setup
apiVersion: batch/v1
kind: Job
metadata:
  name: etl-pipeline
spec:
  template:
    spec:
      initContainers:
      # Wait for dependencies
      - name: wait-for-source
        image: busybox
        command: ['sh', '-c', 'until nc -z source-db 5432; do sleep 2; done']
      # Pre-flight checks
      - name: validate-config
        image: etl:latest
        command: ['./validate-config.sh']
      
      containers:
      # Main ETL process
      - name: etl
        image: etl:latest
        command: ['./run-etl.sh']
      # Monitoring sidecar
      - name: metrics-exporter
        image: prom/statsd-exporter:latest
      
      restartPolicy: Never
```

### Job with Retry and Notification

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: payment-reconciliation
spec:
  backoffLimit: 3
  template:
    spec:
      restartPolicy: OnFailure
      containers:
      - name: reconcile
        image: payment-service:latest
        command:
        - sh
        - -c
        - |
          set -e
          ./reconcile --date=$(date +%Y-%m-%d)
          
          # Notify on success
          curl -X POST $SLACK_WEBHOOK \
            -d '{"text":"Payment reconciliation completed successfully!"}'
        
        env:
        - name: SLACK_WEBHOOK
          valueFrom:
            secretKeyRef:
              name: notifications
              key: slack-webhook
        
        lifecycle:
          preStop:
            exec:
              command:
              - sh
              - -c
              - |
                # Only runs if exit code is 0
                if [ $? -ne 0 ]; then
                  curl -X POST $SLACK_WEBHOOK \
                    -d '{"text":"Payment reconciliation FAILED! Check logs."}'
                fi
      restartPolicy: Never
```

### Batch Processing with ConfigMap Input

```yaml
# ConfigMap with batch parameters
apiVersion: v1
kind: ConfigMap
metadata:
  name: batch-config
data:
  start-date: "2026-01-01"
  end-date: "2026-01-31"
  batch-size: "1000"
---
apiVersion: batch/v1
kind: Job
metadata:
  name: monthly-report
  generateName: monthly-report-  # Auto-generate unique name
spec:
  template:
    spec:
      restartPolicy: OnFailure
      containers:
      - name: report-generator
        image: reports:latest
        envFrom:
        - configMapRef:
            name: batch-config
        command: ['./generate-report.sh']
```

## CronJob Lifecycle

```
CronJob: db-backup (schedule: "0 2 * * *")
     │
     ├─→ 2:00 AM - CronJob controller creates Job
     │       Job name: db-backup-27543600 (timestamp-based)
     │
     ├─→ Job creates Pod
     │       Pod name: db-backup-27543600-xxxxx
     │
     ├─→ Pod runs backup
     │       Connects to postgres
     │       Dumps database
     │       Uploads to S3
     │       Exits 0
     │
     ├─→ Job status: Complete
     │
     ├─→ History management:
     │       Keep last 1 successful job (successfulJobsHistoryLimit: 1)
     │       Keep last 3 failed jobs (failedJobsHistoryLimit: 3)
     │       Old jobs cleaned up
     │
     └─→ 2:00 AM next day → Repeat
```

## Operations and Debugging

```bash
# View jobs
kubectl get jobs
kubectl get jobs -A

# View CronJobs
kubectl get cronjobs

# Check CronJob next run time
kubectl get cronjob db-backup
# Look at: LAST SCHEDULE and NEXT SCHEDULE

# Manually trigger CronJob
kubectl create job --from=cronjob/db-backup manual-backup-001

# View job pods
kubectl get pods -l job-name=db-backup-27543600

# View logs
kubectl logs job/db-backup-27543600

# Follow logs
kubectl logs -f job/db-backup-27543600 --tail=50

# Delete completed jobs (manual cleanup)
kubectl delete job db-backup-27543600

# Delete all completed jobs in namespace
kubectl delete jobs --field-selector status.successful=1

# Suspend CronJob temporarily
kubectl patch cronjob db-backup -p '{"spec": {"suspend": true}}'

# Resume
kubectl patch cronjob db-backup -p '{"spec": {"suspend": false}}'
```

## Best Practices

### 1. Always Set Timeouts

```yaml
spec:
  activeDeadlineSeconds: 3600  # 1 hour max
  backoffLimit: 3
```

### 2. Use TTL for Auto-cleanup

```yaml
spec:
  ttlSecondsAfterFinished: 86400  # Delete after 1 day
```

### 3. Set Resource Limits

```yaml
containers:
- resources:
    requests:
      cpu: 200m
      memory: 256Mi
    limits:
      cpu: "1"
      memory: 1Gi
```

### 4. Idempotency

Design jobs to be re-runnable safely:
```bash
# BAD: Fails if run twice (table already exists)
CREATE TABLE users ...

# GOOD: Idempotent
CREATE TABLE IF NOT EXISTS users ...

# Use database migrations that track state
./migrate --check-current-version
./migrate --direction=up --until=20260101
```

### 5. Proper Exit Codes

```bash
#!/bin/bash
set -e  # Exit on any error

# Your job logic here
./process-data.sh

# Exit 0 on success (important!)
exit 0
```

### 6. Distinguish Retryable vs Fatal Errors

```yaml
spec:
  restartPolicy: Never  # Use Never for fatal errors
  # vs
  restartPolicy: OnFailure  # Use OnFailure for retryable errors
  backoffLimit: 5
```

