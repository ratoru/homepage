---
title: "How to make git status run faster"
description: "Is your starship prompt taking forever to render? Try these git settings!"
publishDate: 2026-03-25
tags: ["devx"]
---

Last week I finally couldn't take waiting for my `starship` prompt to render anymore.

After running `starship timings` I found that `git status` was taking over 150 ms to run each time!
Thankfully, there are two git settings that can speed this up. After enabling `fsmonitor` and
`untrackedCache` my git status time went down to 50 ms.

To enable them, just run these two commands.

```bash
git config --global core.fsmonitor true
git config --global core.untrackedCache true
```

Voilà! You've significantly sped up `git status`(and anything that calls it — starship, lazygit, etc.) in large repos.

## Settings in Detail

**core.fsmonitor**

- Starts a background daemon that hooks into OS filesystem events (FSEvents on
  macOS) to track what files changed. Instead of walking the entire working tree
  on every `git status`, git just asks the daemon "what changed since last time?"
- Requires git 2.36+.
- Only works on local filesystems (not NFS, Docker mounts, etc.).
- Starts lazily per-repo on first git command — no manual intervention needed.

**core.untrackedCache**

- Caches the list of untracked files between runs so git doesn't rescan the tree
  every time.
- Works best in combination with fsmonitor.
- The cache is invalidated automatically when files change.

## Checking & Starting the Daemon

To make sure the filesystem monitor is actually running, you can check its status.
In my case, I had to manually start it for some reason.

```bash
git fsmonitor--daemon status   # check if running in current repo
git fsmonitor--daemon start    # manually start if needed
git status                     # run once to warm the cache after starting
```
