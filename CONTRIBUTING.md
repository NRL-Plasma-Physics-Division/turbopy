Contributing to turboPy
=======================

Thank you for wanting to contribute to turboPy!

Issues
------

Please make use of the Issue Labels when creating issues. This helps us track
and follow the improvements being made to turboPy.

Templates
----------

Use the appropriate template when creating an issue or pull request. Adhere
to the formatting guides within the template to ensure organizational standards
are kept consistent throughout the repo.

Naming Branches
---------------

The naming convention for development branches follows a simple formula:  
`xx-brief-description`  
where `xx = issue ID number` that the branch is addressing. The brief description
should only be a two to three words. Each word should be separated by a hyphen.

Testing
-------

It is our goal to have tests written for all the python modules in turboPy using
the `pytest` framework. Any python code created should have an accompany test file
in `tests/`

Linting
-------

Within turboPy, we strive to keep our code readable and inline with the PEP8
standards. We highly recommend using `pylint` or something similar to check
any code.  
As noted in the README.md, if using `pylint`, add `variable-rgx=[a-z0-9_]{1,30}$`
to your .pylintrc file to allow single character variable names.



