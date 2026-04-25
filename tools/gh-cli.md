# gh (GitHub CLI)

**What:** GitHub's official command-line tool. We use it to create the public repo and push.

**Repo:** https://github.com/cli/cli
**Install:** https://github.com/cli/cli/releases (winget: `winget install --id GitHub.cli`)

## Auth check

```bash
gh auth status
# Should show: ✓ Logged in to github.com account RB112278
```

If not logged in:
```bash
gh auth login -h github.com -p https -s repo,workflow,delete_repo,gist,read:org -w
```

## Commands we used

### Create repo + push in one shot

```bash
git branch -M main           # rename master → main
gh repo create alexa-rb-comp \
  --public \
  --source=. \
  --remote=origin \
  --description "..." \
  --push
```

This created **https://github.com/RB112278/alexa-rb-comp** and pushed `main`.

### Subsequent updates

```bash
git add -A
git commit -m "..."
git push
```

### Make repo private later

```bash
gh repo edit RB112278/alexa-rb-comp --visibility private --accept-visibility-change-consequences
```

### Add a collaborator (so cousin doesn't need a fork)

```bash
gh api -X PUT "repos/RB112278/alexa-rb-comp/collaborators/COUSIN_GITHUB_USERNAME" \
  -f permission=push
```

### Delete the repo (if you change your mind)

```bash
gh repo delete RB112278/alexa-rb-comp --yes
```

## Gotchas

- **`gh auth status` token scopes** must include `repo` for `gh repo create`. Already present in this account.
- **`--source=.` requires being inside the git repo** — `cd` to the project dir first.
- **`--push` flag** auto-pushes the current branch, but the branch must exist locally first (we ran `git branch -M main` to rename master).
