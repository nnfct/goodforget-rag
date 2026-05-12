# GitHub 업로드 절차

이 저장소는 가능하면 Codex가 직접 생성하고 push합니다. 직접 push가 필요한 경우에는 아래 명령을 사용하면 됩니다.

```bash
cd /mnt/c/Users/nesik/Documents/Codex/2026-05-12/windows-terminal
git init
git add .
git commit -m "Initial GoodForget-RAG prototype"
git branch -M main
git remote add origin https://github.com/nnfct/goodforget-rag.git
git push -u origin main
```

이미 원격 저장소가 존재하고 비어 있지 않다면 새 브랜치를 사용합니다.

```bash
git checkout -b codex/goodforget-rag-init
git add .
git commit -m "Initial GoodForget-RAG prototype"
git push -u origin codex/goodforget-rag-init
```
