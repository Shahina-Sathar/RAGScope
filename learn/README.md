# Learn Docker, Deployment & Hosting with RAGScope

A hands-on course. You already know how to code. This teaches you how to **package** your code
(Docker), **run several pieces together** (Docker Compose), and **put it on the internet** for free
(hosting), using this project as the example.

## How to use this course

- Do the lessons **in order**. Each one takes 20–60 minutes.
- Every lesson has the same parts: **Why → Concept → Walkthrough → Try it → Break it on purpose → Check yourself → Common errors**.
- **Type the commands yourself.** Don't copy-paste without reading. Compare what you see with the expected output.
- If you get stuck or the output looks different, paste the command and its output into the Claude chat and say which lesson you're on.
- New word? Look it up in the [glossary](glossary.md).

Run commands in your **WSL (Ubuntu) terminal** in VS Code (Terminal → New Terminal), from the project folder,
unless the lesson says otherwise:

```bash
cd /mnt/d/Work/RAGScope
```

## Lessons (tick them off as you go)

- [ ] [00: The big picture](00-big-picture.md). What "deploying" actually means.
- [ ] [01: Run RAGScope on your laptop](01-run-locally.md). Processes, ports, environment variables.
- [ ] [02: Docker first steps](02-docker-first-steps.md). Images, containers, your first Dockerfile.
- [ ] [03: Dockerize RAGScope](03-dockerize-ragscope.md). The real Dockerfiles, line by line.
- [ ] [04: Docker Compose](04-docker-compose.md). Database, backend and frontend together.
- [ ] [05: Getting ready for production](05-ready-for-production.md). What changes on someone else's computer.
- [ ] [06: A database in the cloud (Neon)](06-database-neon.md)
- [ ] [07: Go live on Hugging Face Spaces](07-hosting-hf-spaces.md)
- [ ] [08: Operate and debug a live app](08-operate-and-debug.md)
- [ ] [09: Where to go next](09-next-steps.md) (optional)

The short version of the deployment steps is in [deploy/README.md](../deploy/README.md).
Use it as a cheat sheet once you understand the lessons.
