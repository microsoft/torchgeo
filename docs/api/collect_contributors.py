from git import Repo

# Specify the path to the local directory of the torchgeo repository
repo = Repo('.')

repo.git.fetch('origin')

# Checkout the branch
repo.git.checkout('releases/v0.5')

# Get the list of contributors to the release
contributors = set()
for commit in repo.iter_commits():
    contributors.add(commit.author.name)

# Print out the list of contributors
for contributor in contributors:
    print(contributor)
