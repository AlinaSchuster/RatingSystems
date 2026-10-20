# Rules for AI agents (Claude, Copilot, Cline, ...)

## Deleting needs Alina's written OK

Never delete anything in this repository that has not been discussed first, and only after Alina has given her OK **in writing in the chat**.

A permission dialog or an "allow" / "approve" click does **not** count. Alina often works in auto mode, where those get approved without being read.

This covers, among others:
- `rm`, `del`, `rmdir`, deleting files or folders in any way
- overwriting or emptying an existing file instead of editing it
- `git rm`, `git clean`, `git reset --hard`, `git checkout -- <file>`, `git restore`, `git stash drop`
- force pushes, rewriting history, deleting branches or tags
- moving files out of the repository

If something looks like it should go, list it and ask. Do not delete it "to tidy up".

## Secrets

Never commit, print, paste or move secrets (`.env`, API tokens such as the Notion token). They stay on Alina's machine.

*(Established 2026-10-20.)*
