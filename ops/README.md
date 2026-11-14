# Operations

Tools for running long jobs unattended on an always-on Mac (training, teacher audio, checks).

| File | What it does |
|---|---|
| `jobqueue.sh` | Runs queued jobs one at a time, forever. Jobs are shell scripts in `~/samleng-jobs/pending/`, run in name order; the current one sits in `running/` and is run again after a restart, so **every job must be able to resume**. A failure moves the job to `failed/`, writes `~/samleng-jobs/PAUSED` and stops the queue until that file is removed. Optional phone alerts through [ntfy](https://ntfy.sh) (topic read from `~/samleng-ops/ntfy_topic`, never stored in the repo). |
| `com.samleng.jobqueue.plist` | LaunchAgent that starts the queue at login and restarts it if it exits (install steps in the file). |
| `watch_run.sh` | Started by a training job: scores every 2,000th checkpoint with `scripts/eval_checkpoint.py` while training runs, sends a progress update, and keeps a copy of each scored checkpoint (training keeps only its last three). |
| `site_publish.sh` | Every 5 minutes (LaunchAgent `com.samleng.sitestatus.plist`): writes the project page's live status with `scripts/site_status.py` and uploads it to a public gist. |
| `notebook.jsonl` | Lab notebook: one JSON line per finished experiment (`id`, `date`, `kind`, `question`, `setup`, `result`, `verdict`, `next`). |

## Queue in practice

```bash
# add a job (it must resume: --resume, --continue-run, cached results ...)
cp my-job.sh ~/samleng-jobs/pending/070-my-job.sh

tail ~/samleng-jobs/queue.log            # start / done / FAILED lines
tail ~/samleng-jobs/logs/070-my-job.log  # the job's own output

# after fixing a failed job
mv ~/samleng-jobs/failed/070-my-job.sh ~/samleng-jobs/pending/ && rm ~/samleng-jobs/PAUSED
```

The queue never starts a job while another GPU job is running, including ones started by hand.

The project page shows what the running job is doing from optional lines at the top of its script:

```bash
# site-kind: testing            # training is detected from the job's own log; else testing or recording
# site-run: distill_v2_clear    # whose scores the page shows
# site-en: Testing how much slowing the voice down costs.
# site-km: <the same sentence in Khmer>
# site-progress: outputs/train/eval/run*_seed*/*/results.csv 17   # files done out of a total
```
