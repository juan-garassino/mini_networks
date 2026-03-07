# Adding a Dataset

This guide shows how to add a new dataset to the shared registry so every model and composition can use it consistently.

**1) Add a dataset entry**
Edit `src/mini_networks/core/data/registry.py` and add a new `name` in `get_dataset()`:
- If the dataset is vision‑based, decide which task modes make sense (`classification`, `binary_segmentation`, `multiclass_segmentation`, `detection`, `clip`).
- If the dataset is text‑based, decide whether it should be `text_file`‑style or a named dataset like `tiny_shakespeare`.

**2) Add task‑specific wrappers if needed**
If the dataset needs derived targets (segmentation masks, detection boxes, etc.), create a small dataset wrapper class in the same file.
- Keep it simple and deterministic.
- Use the same return signatures as existing modes so models remain interchangeable.

**3) Use the registry from models**
Make sure models use `get_dataloader()` so they inherit registry behavior.
- This avoids one‑off dataset logic inside trainers.
- It ensures consistent `fast_demo`, batch sizing, and caching.

**4) Document it**
Add the dataset to `docs/datasets.md` under the appropriate section.

**5) Add a smoke test**
Add a minimal loader test to `tests/test_smoke_registry.py`:
- 1 batch, `fast_demo=True`, CPU only.
- Just assert that a batch is returned and has the expected structure.

**Checklist**
- `get_dataset()` entry exists
- Task modes return consistent shapes
- Models use `get_dataloader()`
- Docs updated
- Smoke test added
