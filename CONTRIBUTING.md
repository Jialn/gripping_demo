1. Install code style tools
```bash
pip install pre-commit cpplint pydocstyle
sudo apt install clang-format
```

2. At your local repo root, run
```bash
pre-commit install
```

3. Make local changes
```bash
git co -b PR_change_name origin/master
```

 Make change to your code and test

 Optional checking without commit
```bash
pre-commit run --files ./*
```

4. Make pull request:
```bash
git push origin PR_change_name
```
