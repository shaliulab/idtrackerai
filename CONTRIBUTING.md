# Contributing to idtracker.ai

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

The following is a set of guidelines for contributing to idtracker.ai. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

#### Table Of Contents

[Code of Conduct](#code-of-conduct)

[I don't want to read this whole thing, I just have a question!!!](#i-dont-want-to-read-this-whole-thing-i-just-have-a-question)

[What should I know before I get started?](#what-should-i-know-before-i-get-started)
  * [idtracker.ai and its repositories](#idtrackerai-and-its-repositories)

[How Can I Contribute?](#how-can-i-contribute)
  * [Reporting Bugs](#reporting-bugs)
  * [Suggesting Enhancements](#suggesting-enhancements)
  * [Your First Code Contribution](#your-first-code-contribution)
  * [Pull Requests](#pull-requests)

## Code of Conduct

This project and everyone participating in it is governed by the [idtracker.ai of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [idtrackerai@gmail.com](mailto:idtrackerai@gmail.com).

## I don't want to read this whole thing I just have a question!!!

Please don't file an issue to ask a question. First, try to find if somebody else asked your question in the [idtracker.ai users group](https://groups.google.com/forum/#!forum/idtrackerai_users).

## What should I know before I get started?

### idtracker.ai and its repositories

idtracker.ai is divided in three different repositories: [the idtracker.ai module](https://gitlab.com/polavieja_lab/idtrackerai), [the idtracker.ai app](https://gitlab.com/polavieja_lab/idtrackerai-app) and the [idtracker.ai validator submodule](https://github.com/UmSenhorQualquer/pythonvideoannotator-module-idtrackerai).

For now all issues can be filed to the [idtracker.ai mudule repository](https://gitlab.com/polavieja_lab/idtrackerai)

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report for idtracker.ai. Following these guidelines helps maintainers and the community understand your report :pencil:, reproduce the behavior :computer: :computer:, and find related reports :mag_right:.

Before creating bug reports, please check [this list](#before-submitting-a-bug-report) as you might find out that you don't need to create one. When you are creating a bug report, please [include as many details as possible](#how-do-i-submit-a-good-bug-report).

> **Note:** If you find a **Closed** issue that seems like it is the same thing that you're experiencing, open a new issue and include a link to the original issue in the body of your new one.

#### Before Submitting A Bug Report

* **Check if you can reproduce the problem [in the latest version of idtracker.ai](LINK TO HOW TO UPDATE)**.
* **Check the [FAQs on the webpage](https://idtracker.ai/en/latest/FAQs.html) and the [idtracker.ai users group](https://groups.google.com/forum/#!forum/idtrackerai_users)** for a list of common questions and problems.
* **Determine [which repository the problem should be reported in](#idtracker.ai-and-its-repositories)**.
* **Perform a [cursory search](https://gitlab.com/polavieja_lab/idtrackerai/issues)** to see if the problem has already been reported. If it has **and the issue is still open**, add a comment to the existing issue instead of opening a new one.

#### How Do I Submit A (Good) Bug Report?

Bugs are tracked as [GitLab issues](https://docs.gitlab.com/ee/user/project/issues/).

Explain the problem and include additional details to help maintainers reproduce the problem:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact steps which reproduce the problem** in as many details as possible. For example, start by explaining how you started idtracker.ai, e.g. which command exactly you used in the terminal, or how you started idtracker.ai otherwise. When listing steps, **don't just say what you did, but explain how you did it**.
* **Provide specific examples to demonstrate the steps**. Include links to filess, or copy/pasteable snippets, which you use in those examples. If you're providing snippets in the issue, use [Markdown code blocks](https://help.github.com/articles/markdown-basics/#multiple-lines).
* **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
* **Explain which behavior you expected to see instead and why.**
* **Include screenshots and animated GIFs** which show you following the described steps and clearly demonstrate the problem. You can use [this tool](https://www.cockos.com/licecap/) to record GIFs on macOS and Windows, and [this tool](https://github.com/colinkeenan/silentcast) or [this tool](https://github.com/GNOME/byzanz) on Linux.
* **If you're reporting that idtracker.ai crashed**, include a crash report with a stack trace from the operating system. Copy and paste the output of the terminal, or add a screenshot.
* **If the problem wasn't triggered by a specific action**, describe what you were doing before the problem happened and share more information using the guidelines below.

Provide more context by answering these questions:

* **Did the problem start happening recently** (e.g. after updating to a new version of idtracker.ai) or was this always a problem?
* If the problem started happening recently, **can you reproduce the problem in an older version of idtracker.ai?** What's the most recent version in which the problem doesn't happen? You can download older versions of idtracker.ai from [the releases page](https://gitlab.com/polavieja_lab/idtrackerai) or install them from the [PyPI package repository](https://pypi.org/project/idtrackerai/3.0.13a0/).
* **Can you reliably reproduce the issue?** If not, provide details about how often the problem happens and under which conditions it normally happens.
* If the problem is related to working with files (e.g. opening and editing files), **does the problem happen for all files and projects or only some?**.

Include details about your configuration and environment:

* **Which version of idtracker.ai are you using?** You can get the exact version by running `import idtrackerai` and `idtrackerai.__version__` inside of the Python command line.
* **What's the name and version of the OS you're using**?
* **Are you running idtracker.ai in GUI mode or in terminal mode**?
* **Are you using [local settings files](https://idtracker.ai/en/latest/advanced_parameters.html)** `local_settings.py`. If so, provide the contents of those files, preferably in a [code block](https://help.github.com/articles/markdown-basics/#multiple-lines) or with a link to a [gist](https://gist.github.com/).

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for idtracker.ai, including completely new features and minor improvements to existing functionality. Following these guidelines helps maintainers and the community understand your suggestion :pencil: and find related suggestions :mag_right:.

Before creating enhancement suggestions, please check [this list](#before-submitting-an-enhancement-suggestion) as you might find out that you don't need to create one. When you are creating an enhancement suggestion, please [include as many details as possible](#how-do-i-submit-a-good-enhancement-suggestion), including the steps that you imagine you would take if the feature you're requesting existed.

#### Before Submitting An Enhancement Suggestion

* **heck if you're using [the latest version of idtracker.ai](LINK TO HOW TO UPDATE)**.
* **Determine [which repository the enhancement should be suggested in](#idtrackerai-and-its-repositories).**
* **Perform a [cursory search](https://gitlab.com/polavieja_lab/idtrackerai/issues)** to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.

#### How Do I Submit A (Good) Enhancement Suggestion?

Enhancement suggestions are tracked as [GitLab issues](https://docs.gitlab.com/ee/user/project/issues/). After you've determined [which repository](#idtrackerai-and-its-repositories) your enhancement suggestion is related to, create an issue on that repository and provide the following information:

* **Use a clear and descriptive title** for the issue to identify the suggestion.
* **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
* **Describe the current behavior** and **explain which behavior you expected to see instead** and why.
* **Include screenshots and animated GIFs** which help you demonstrate the steps or point out the part of idtracker.ai which the suggestion is related to. You can use [this tool](https://www.cockos.com/licecap/) to record GIFs on macOS and Windows, and [this tool](https://github.com/colinkeenan/silentcast) or [this tool](https://github.com/GNOME/byzanz) on Linux.
* **Explain why this enhancement would be useful** to most idtracker.ai users.
* **Specify which version of idtracker.ai you're using.** You can get the exact version by running `import idtrackerai` and `idtrackerai.__version__` inside of the Python command line.
* **Specify the name and version of the OS you're using.**

### Your First Code Contribution

Unsure where to begin contributing to idtracker.ai? You can start by looking through these `beginner` and `help-wanted` issues:

* [Beginner issues][beginner] - issues which should only require a few lines of code, and a test or two.
* [Help wanted issues][help-wanted] - issues which should be a bit more involved than `beginner` issues.

Both issue lists are sorted by total number of comments. While not perfect, number of comments is a reasonable proxy for impact a given change will have.

### Pull Requests

...
