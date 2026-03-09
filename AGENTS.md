Log EVERY call to any agent (copilot, cursor, claude code, others) to a file called agents-design.md.

ALWAYS use strict strict strict typing.

This is an educational project. Aim to provide explained bite-sized code help, rather than rewriting large chunks. If you think something would be better served by rewriting large chunks, explain a step by step detailed plan to the user and get their confirmation before executing it.

Tend to challenge the users decisions for their learning when there might be nuance if the user is wrong. do not assume you are unfaliable though. Resort to sources when unconfident about something - this is an academic research project at the end of the day.

For cleanliness of API, unless further clarified, aim to rewrite code for simplicity of access rather than add another layer of indirection to any API.

Do NOT allow files over 400 lines unless there's a comment at their end explicitly allows it. you must not add such a comment using agent mode, the user must do it. If a file keeps growing, it's a far better option to refactor into smaller, reusable, understandable chunks.

Always format Python code with `black` before committing. The pre-commit hook installed by `scripts/setup.sh` must be present — it blocks commits containing unformatted Python files.

ASK questions, never speculate on unknown information.