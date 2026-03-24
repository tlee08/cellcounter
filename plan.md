Looking at your `Pipeline` class, there's a design smell here. You're using **class as a namespace with mutable state**:

```python
class Pipeline:
    _gpu_cluster = LocalCUDACluster  # mutable class state
    cellc_funcs = GpuCellcFuncs      # mutable class state

    @classmethod
    def set_gpu(cls, *, enabled: bool = True) -> None:
        cls._gpu_cluster = ...  # mutates class state
```

## Problems with current approach

1. **Hidden mutable state** - `set_gpu()` silently changes global behavior
2. **Testing is hard** - class state persists between tests
3. **Concurrent pipelines impossible** - can't run GPU and CPU pipelines simultaneously
4. **Not really OOP** - it's just a bag of functions with shared config

## Recommended: Instance-based with config injection

```python
@dataclass
class PipelineConfig:
    gpu_enabled: bool = True
    heavy_n_workers: int = 2
    busy_n_workers: int = 6

    @property
    def cellc_funcs(self) -> type[CpuCellcFuncs]:
        return GpuCellcFuncs if self.gpu_enabled else CpuCellcFuncs

    def gpu_cluster(self) -> SpecCluster:
        if self.gpu_enabled:
            return LocalCUDACluster()
        return LocalCluster(n_workers=self.heavy_n_workers, threads_per_worker=1)


class Pipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

    def heavy_cluster(self) -> SpecCluster:
        return LocalCluster(n_workers=self.config.heavy_n_workers, ...)

    def cellc1(self, proj_dir: Path, *, overwrite: bool = False, tuning: bool = False) -> None:
        with cluster_process(self.config.gpu_cluster()):
            ...
            bgrm_arr = da.map_blocks(self.config.cellc_funcs.tophat_filt, ...)
```

## Benefits

| Current                                        | Instance-based                                                   |
| ---------------------------------------------- | ---------------------------------------------------------------- |
| `Pipeline.set_gpu(False)` mutates global state | `Pipeline(config=PipelineConfig(gpu_enabled=False))` is explicit |
| Tests can't be isolated                        | Each test gets fresh instance                                    |
| Can't run parallel pipelines                   | Multiple `Pipeline` instances with different configs             |
| Decorators inspect class state                 | Decorators can be simpler or methods                             |

---

**Should I make this refactor?** It's a significant change (~50 method signatures), but it would make the codebase more testable and eliminate the surprising mutable class state. The decorator `@_check_overwrite` would need adjustment too (it currently reads from `Pipeline._get_pfm`).
