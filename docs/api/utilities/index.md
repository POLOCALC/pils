# Utilities

Helper modules and decoders.

## Modules

| Module | Description |
|--------|-------------|
| [Tools](tools.md) | File handling, time conversion, math utilities |
| [Decoders](decoders.md) | Binary format decoders |

## Import

```python
from pils.utils.tools import (
    get_files_by_extension,
    search_in_folder,
    unix_to_datetime,
    datetime_to_unix,
)
from pils.decoders import decode_kernel_data
```

## Quick Start

```python
from pils.utils.tools import get_files_by_extension
from pathlib import Path

# Find all CSV files
csv_files = get_files_by_extension(
    folder=Path("/data/flight"),
    extension=".csv"
)

for f in csv_files:
    print(f.name)
```

---

## See Also

- [Tools](tools.md) - Utility functions
- [Decoders](decoders.md) - Binary decoders
