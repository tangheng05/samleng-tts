#!/bin/zsh
# Run queued GPU jobs one at a time, forever; meant to be started by launchd (ops/com.samleng.jobqueue.plist).
#
#   ~/samleng-jobs/pending/NNN-name.sh   queued, run in name order
#   ~/samleng-jobs/running/              the current job; re-run after a restart, so jobs must resume
#   ~/samleng-jobs/done/ failed/         finished jobs; a failure writes PAUSED and stops the queue
#   ~/samleng-jobs/logs/NNN-name.log     each job's output; queue.log has start/end lines
#
# Nothing starts while another GPU job runs (also ones started by hand). Remove PAUSED to resume.

export PATH=$HOME/.local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin
export LC_ALL=en_US.UTF-8
Q=${SAMLENG_JOBS:-$HOME/samleng-jobs}
mkdir -p $Q/{pending,running,done,failed,logs}

busy() { pgrep -f "train_vits/train.py|python train.py|tts_voxcpm/synthesize.py|distill_prepare.py" >/dev/null }
log() { print -r -- "$(date '+%F %T') $*" >> $Q/queue.log }
# Phone alert via ntfy; the private topic lives in ~/samleng-ops/ntfy_topic (never in the repo).
notify() {
  local topic=$(cat ~/samleng-ops/ntfy_topic 2>/dev/null)
  [[ -n $topic ]] && curl -s -m 15 -o /dev/null -H "Title: $1" -H "Priority: ${3:-default}" -H "Tags: ${4:-gear}" -d "$2" "https://ntfy.sh/$topic"
}

log "runner started (pid $$)"
while true; do
  if [[ -f $Q/PAUSED ]] || busy; then
    sleep 60
    continue
  fi
  job=($Q/running/*.sh(N))
  if (( ! $#job )); then
    next=($Q/pending/*.sh(N))
    if (( ! $#next )); then
      sleep 60
      continue
    fi
    mv $next[1] $Q/running/
    job=($Q/running/${next[1]:t})
  fi
  j=$job[1]
  name=${j:t:r}
  log "start $name"
  notify "Started: $name" "$(head -1 $j | sed 's/^# //')" low
  caffeinate -is /bin/zsh $j >> $Q/logs/$name.log 2>&1
  rc=$?
  if (( rc == 0 )); then
    mv $j $Q/done/
    log "done $name"
    next_job=$(ls $Q/pending 2>/dev/null | head -1)
    notify "Finished: $name" "Next: ${next_job:-nothing queued}" default white_check_mark
  else
    mv $j $Q/failed/
    print -r -- "$name exited $rc at $(date '+%F %T')" > $Q/PAUSED
    log "FAILED $name (exit $rc), queue paused"
    notify "FAILED: $name" "Exit $rc. The queue is paused; ask Claude to check. Last log line: $(tail -1 $Q/logs/$name.log | cut -c1-200)" high rotating_light
  fi
done
