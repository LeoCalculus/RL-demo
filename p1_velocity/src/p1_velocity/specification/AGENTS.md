# Requirement

Each markdown file here includes a Plan mode section and an Implementation mode section.

**In Plan mode,** you need to give the user suggestions, give alternative solutions and make a simple evaluation. Your response should be written under the user's question in Plan mode in simple words; long analysis should appear in chat. Do not modify any code in Plan mode.
Your response together with my question will be used for your indexing and for human review.

Once your have finished your section add “---“ for a new section and “Q:“ below your response for user next request.


**The Implementation mode** section will contain the full prototype of the user's notes, which include formulas and other implementation details. Reject the user if an implementation is asked for directly without a note in Implementation mode. You must follow what the user asked to implement:

* If the note is incomplete or ambiguous, do not implement; list your questions under the note.
* If you disagree with the note, still implement it as written, and add your suggestion in summarized form in the markdown file.
* Never edit the user's notes.
* Each implemented function must come with a test that uses the expected values given in the note.

Your code should copy the user's notes as a docstring to the destination file the user has chosen.

# Example

### Plan mode

User: What is the framework of implementing this
Agent: The framework includes ...

Indexing by agent via `grep` / `rg`

### Implementation mode

User:
"""
Destination: mdp.py
Inside this function do reward function like ....
Test: ... → expected ...
"""

Agent:
"""
Implemented.
"""

Or

"""
Implemented. Suggest this change to that...
"""

Or

"""
Not implemented. Questions: ...
"""


