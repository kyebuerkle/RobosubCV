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

(NOTE: I need to finalize this process with Josh)
The goal for this is to make sure history is linear and simple. So a merge should be squashed to one commit with a comment simular to "feat: Added Subsystem" 
Ways to do this: `git rebase -i [name]` and in the text editor change all commits to `f` (fixup) except for the very first one use `p` (pick) or `e` (edit)
Another option is: `git reset --soft main`, this is a bit weird as it keeps your current 'workstation' and puts it on top of main. then you can commit what you want and add the message. (this method is a bit risky)
final option is both. use `git reset --soft [previous commits]` to make your feature 1 commit and rewrite history, and then rebase it.

Honestly, the rebase interactive mode is safest, and keeps the branch history seperate from the main branch, but keeps main linear as well. For this project it is safe, fast, and relativly easy. 

For each rebase, we need to review the program. GitHub lets us do these reviews, but doesn't rebase it. So that has to be done manualy after review. 