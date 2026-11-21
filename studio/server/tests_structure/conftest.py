"""Structure gate: rules about the C++/CUDA/Metal tree that no runtime test enforces.

These are not behavior tests. Each one reads another language's source or CMake
file and asserts that a convention still holds: which headers may be included,
which environment variables may be read, which format families are sharded, and
so on. They live outside ``tests/`` on purpose.

* ``pytest studio/server/tests`` runs the Python suite and never depends on the
  C++/CUDA/Metal text, so refactoring native code cannot break it;
* ``pytest studio/server/tests_structure`` runs this gate.

When a rule here describes runtime behavior instead of a textual convention,
move it into a native test under ``csrc/tests/`` and delete the copy here.
"""
