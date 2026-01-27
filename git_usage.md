#		Git Usage

This markdown goes through how our team will use Git for a better history and saving.

##		Summary

Summary of git usage:
The goal is to keep our history linear on the main / master branch. Every push to this branch requires review from at least 1 other teammate. 
1. Fetch / Pull changes: `git fetch origin` or `git pull`
   1. Always update your main branch to make sure any changes transfer from the remote repository
   2. when fetching, makes sure your changes merge by rebasing or commiting
2. Make a new branch: `git branch [name]`
   1. Every feature needs a seperate branch for review
   2. Switch to branch `git switch [name]` or `git checkout [name]`
   3. to make and switch: `git checkout -b [name]`
3. Stream branch to remote repository: `git push -u origin [name]` or `git push --set-upstream origin [name]`
   1. Puts branch in GitHub, double check from browser. 
   2. After the branch is set-upstream, just use `git push origin [name]` to push any more commits
4. Make sure to push after every programing session
   1. you never know when something will happen to your computer, so it's good to keep it saved in GitHub

###		Push a change for review

On the `main` branch, we want to keep a linear history. To do this there can't be any merging (unfortunatly GitHub makes us pay to add rulesets that automatically ensure this, so we'll have to be extra carefull).

Once a branch is ready for review:
1. Push your branch: `git push origin [name]`
2. Select "Compare & Pull Request" in GitHub
3. Ask for review 
   1. Add a reviewer (Kye can do most)
   2. Add yourself as the signee
   3. press "Pull Request" to request the review

After the program has been reviewed, rebase to main:
1. press the arrow next to "Merge" 
   1. Select "Squash and Rebase"

COMMAD LINE MODE:
1. Pull the current branch after review and edit
2. Switch to main: `git checkout main`
   1. also make sure this pulled from the remote
3. rebase interactive mode: `git rebase -i [name]`
4. Top commit use `p` (pick), or `e` (edit)
   1. Change message to include the feature
   2. Ex: "feat: Training script for YOLOv8 model"
5. All other commits type `f` (fixup) or `s` (squash)
   1. squash keeps all the messages, fixup doesn't
6. Save and exit text editor
   1. verifiy rebase worked: `git log -5`
7. Push changes: `git push origin main`
   1. or just: `git push`

###		Re-writing History

In git, you can re-write the history of your branch. You can do this to make each commit more meaningful and organized even if it's not chronologically correct. You can do this to help simplify your branch before rebasing, or make the whole feature in one commit without squashing.

Some key re-writing commands:
- `git rebase -i`
  - rebase your own branch and squash or edit different commits
- `git reset --hard [commit]`
  - resets to a previous commit, changes entire workshop to that commit
  - workshop is the current files you can see (`git status` menu)
  - this is safe, you can go back to previous commits and 'undo' resets from remote
  - instead of `[commit]` you can also put in a branch name
- `git reset --soft [commit]`
  - resets to a pervious commit, keeps entire workshop
  - workshop is the current files you can see (`git status` menu)
  - this is not safe, as going back will have future files in workshop, this command is the one that truely re-writes the history
    - Needs a `git push --force` to actually merge on remote because it is unsafe
  - instead of `[commit]` you can also put in a branch name