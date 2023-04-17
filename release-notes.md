Adaptive-Scheduler v2.0.0 (beta) 🚀

I am excited to announce the most significant update in 4 years for Adaptive-Scheduler.
I have been working insane hours Friday and this weekend (I got too excited 😂) to get it ready.
Adaptive-scheduder got to a state where I was afraid to make any changes, because there were no tests.
So I added many tests, and now I am confident to make changes and add new features (which I did).

I am looking for any beta testers!

Key features and improvements:

*   Native support for `concurrent.futuresProcessPoolExecutor`
*   Function calls can now be submitted to workers ***without pickling***
*   Fully typed code using mypy for better code quality and reliability
*   Numerous bug fixes identified through tests and mypy
*   Added an 391 tests, 100% testing coverage in core files
*   Major code reorganization: **7243** line changes and **3632** new lines of code
*   Removed `run_script.py` and templated code; replaced with a launcher script, simplifying job script creation
*   Updated README.md and documentation

Note that I haven't tagged a new release yet, so you will need to install from the `main` branch, with:
`pip install -U https://github.com/basnijholt/adaptive-scheduler/archive/refs/heads/main.zip`
