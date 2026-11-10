#!/bin/zsh
# One unattended operator check (started every 2 h by launchd: com.samleng.operator).
export PATH=$HOME/.local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin
export LC_ALL=en_US.UTF-8
cd ~/code/samleng-tts
mkdir -p ~/samleng-ops/runs
claude -p "Do your scheduled operator check now, following your instructions." \
  --append-system-prompt-file ~/samleng-ops/OPERATOR.md \
  --settings ~/samleng-ops/settings.json \
  --permission-mode dontAsk --permission-prompts none \
  --add-dir ~/samleng-jobs ~/samleng-ops \
  --model sonnet \
  < /dev/null > ~/samleng-ops/runs/$(date +%Y%m%d-%H%M).log 2>&1

# Phone alert when the operator flags something new for the manager (not repeated every run).
needs=$(sed -n '/NEEDS MANAGER/,$p' ~/samleng-ops/STATUS.md 2>/dev/null | tail -n +2 | grep -viE '^[[:space:]]*-?[[:space:]]*none[[:space:]]*$' | grep -v '^[[:space:]]*$' | head -5)
if [[ -n $needs && $needs != "$(cat ~/samleng-ops/.last_needs 2>/dev/null)" ]]; then
  print -r -- "$needs" > ~/samleng-ops/.last_needs
  topic=$(cat ~/samleng-ops/ntfy_topic 2>/dev/null)
  [[ -n $topic ]] && curl -s -m 15 -o /dev/null -H "Title: Operator needs the manager" -H "Priority: high" -H "Tags: warning" -d "$needs" "https://ntfy.sh/$topic"
elif [[ -z $needs ]]; then
  rm -f ~/samleng-ops/.last_needs
fi
